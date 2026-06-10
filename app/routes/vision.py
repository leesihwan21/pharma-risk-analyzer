import io
import os
import base64
import pickle
import threading
import cv2
import numpy as np

from PIL import Image
from ultralytics import YOLO
from flask import Blueprint, render_template, jsonify, request, Response, current_app
from app import cache
from app.models import db, PredictionLog
 
vision = Blueprint('vision', __name__)
 
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ml')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'data', 'processed', 'processed_faers.csv')
 
camera = None
camera_lock = threading.Lock()
 
def load_yolo():
    yolo_path = os.path.join(MODEL_DIR, 'best.pt')
    return YOLO(yolo_path)
 
def load_model():
    model = pickle.load(open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb'))
    le_drug = pickle.load(open(os.path.join(MODEL_DIR, 'le_drug.pkl'), 'rb'))
    le_reac = pickle.load(open(os.path.join(MODEL_DIR, 'le_reac.pkl'), 'rb'))
    return model, le_drug, le_reac
 
def load_df():
    import pandas as pd
    return pd.read_csv(DATA_PATH)
 
def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera
 
def generate_frames():
    yolo = load_yolo()
    while True:
        with camera_lock:
            cam = get_camera()
            success, frame = cam.read()
        if not success:
            break
        results = yolo(frame, verbose=False)
        frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
 
@vision.route('/webcam')
def webcam():
    return render_template('webcam.html')
 
@vision.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
 
@vision.route('/api/stop_camera')
def stop_camera():
    global camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
            camera = None
    return jsonify({'message': 'Camera stopped'})
 
@vision.route('/api/detect', methods=['POST'])
def detect_pill():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file required'}), 400
 
    file = request.files['image']
    drug_hint = request.form.get('drugname', '').upper()
    sex = request.form.get('sex', 'F')
    age = float(request.form.get('age', 50))
 
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
 
    yolo = load_yolo()
    results = yolo(img)
 
    detections = []
    detected_drugs = []
 
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo.names[cls].upper()
            detections.append({'label': label, 'confidence': round(conf * 100, 1)})
            detected_drugs.append(label)
 
    detected_drugs = list(set(detected_drugs))
 
    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_img)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
 
    risk_result = None
    combo_result = None
    target_drug = drug_hint if drug_hint else (detected_drugs[0] if detected_drugs else None)
 
    if target_drug:
        try:
            model, le_drug, le_reac = load_model()
            risk_rates = pickle.load(open(os.path.join(MODEL_DIR, 'risk_rates.pkl'), 'rb'))
 
            if target_drug in le_drug.classes_:
                df = load_df()
                result_df = df[df['drugname'].str.upper() == target_drug]
                top_reac = result_df['pt'].value_counts().head(1)
 
                if len(top_reac) > 0:
                    reac = top_reac.index[0]
                    if reac in le_reac.classes_:
                        drug_enc = le_drug.transform([target_drug])[0]
                        reac_enc = le_reac.transform([reac])[0]
                        sex_enc = 0 if sex == 'F' else 1
                        drug_risk_rate = risk_rates['drug_risk'].get(drug_enc, 0.5)
                        reac_risk_rate = risk_rates['reac_risk'].get(reac_enc, 0.5)
                        combo_risk_rate = risk_rates['combo_risk'].get(f"{drug_enc}_{reac_enc}", 0.5)
 
                        X = [[drug_enc, reac_enc, sex_enc, age,
                              drug_risk_rate, reac_risk_rate, combo_risk_rate]]
                        pred = model.predict(X)[0]
                        prob = model.predict_proba(X)[0]
 
                        risk_result = {
                            'drug': target_drug, 'reaction': reac,
                            'risk_label': 'High Risk' if pred == 1 else 'Low Risk',
                            'safe': round(float(prob[0]) * 100, 1),
                            'risk': round(float(prob[1]) * 100, 1)
                        }
 
                        log = PredictionLog(
                            drugname=target_drug, reaction=reac, age=age, sex=sex,
                            risk=int(pred),
                            safe_prob=round(float(prob[0]) * 100, 1),
                            risk_prob=round(float(prob[1]) * 100, 1)
                        )
                        db.session.add(log)
                        db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Single drug risk analysis error: {str(e)}")
 
    if len(detected_drugs) >= 2:
        try:
            model, le_drug, le_reac = load_model()
            risk_rates = pickle.load(open(os.path.join(MODEL_DIR, 'risk_rates.pkl'), 'rb'))
            sex_enc = 0 if sex == 'F' else 1
            combo_temp_results = []
 
            for d in detected_drugs[:2]:
                if d not in le_drug.classes_:
                    continue
                drug_enc = le_drug.transform([d])[0]
                drug_risk_rate = risk_rates['drug_risk'].get(drug_enc, 0.5)
                df = load_df()
                top_reacs = df[df['drugname'].str.upper() == d]['pt'].value_counts().head(3).index.tolist()
 
                drug_results = []
                for reac in top_reacs:
                    if reac not in le_reac.classes_:
                        continue
                    reac_enc = le_reac.transform([reac])[0]
                    reac_risk_rate = risk_rates['reac_risk'].get(reac_enc, 0.5)
                    combo_risk_rate = risk_rates['combo_risk'].get(f"{drug_enc}_{reac_enc}", 0.5)
                    X = [[drug_enc, reac_enc, sex_enc, age,
                          drug_risk_rate, reac_risk_rate, combo_risk_rate]]
                    pred_c = model.predict(X)[0]
                    prob_c = model.predict_proba(X)[0]
                    drug_results.append({
                        'reaction': reac,
                        'risk_label': 'High Risk' if pred_c == 1 else 'Low Risk',
                        'risk_prob': round(float(prob_c[1]) * 100, 1)
                    })
 
                combo_temp_results.append({
                    'drug': d,
                    'drug_risk_rate': round(drug_risk_rate * 100, 1),
                    'reactions': drug_results
                })
 
            if combo_temp_results:
                combo_result = combo_temp_results
        except Exception as e:
            print(f"Combo analysis error: {str(e)}")
 
    return jsonify({
        'detections': detections,
        'image': img_b64,
        'risk_result': risk_result,
        'combo_result': combo_result
    })

@vision.route('/api/detect_and_lookup', methods=['POST'])
def detect_and_lookup():
    from flask import current_app
    import requests as http_requests
    if 'image' not in request.files:
        return jsonify({'error': 'Image file required'}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # YOLOv8 탐지
    yolo = load_yolo()
    results = yolo(img)

    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo.names[cls].upper()
            detections.append({'label': label, 'confidence': round(conf * 100, 1)})

    # 결과 이미지
    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_img)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 내부 drug-lookup API 호출로 식약처 검색
    drug_hint = request.form.get('drugname', '').strip()
    mfds_results = []

    search_terms = []
    if drug_hint:
        search_terms.append(drug_hint)
    else:
        for det in detections[:3]:
            if det['label'].upper() != 'PILL':
                search_terms.append(det['label'])

    for term in search_terms:
        try:
            lookup_res = http_requests.get(
                f'http://localhost:5001/api/drug-lookup?name={term}',
                timeout=10
            )
            data = lookup_res.json()
            if data.get('korean'):
                k = data['korean']
                mfds_results.append({
                    'detected_as': term,
                    'name': k.get('name', '-'),
                    'company': k.get('company', '-'),
                    'shape': k.get('shape', '-'),
                    'color': k.get('color', '-'),
                    'img_url': k.get('img_url', ''),
                    'class_name': k.get('class_name', '-'),
                })
        except:
            pass
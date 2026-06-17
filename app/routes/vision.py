import io
import os
import base64
import pickle
import threading
import cv2
import numpy as np
import re
import sqlite3
import easyocr

from PIL import Image
from ultralytics import YOLO
from flask import Blueprint, render_template, jsonify, request, Response, current_app
from app import cache
from app.models import db, PredictionLog

vision = Blueprint('vision', __name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ml')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'data', 'processed', 'processed_faers.csv')
PILL_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'data', 'pill_identity.db')

camera = None
camera_lock = threading.Lock()


def load_yolo():
    yolo_path = os.path.join(MODEL_DIR, 'best.pt')
    return YOLO(yolo_path)


_ocr_reader = None
def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en', 'ko'], gpu=False)
    return _ocr_reader


def crop_and_preprocess(img_pil, box, idx=0, padding=0.15):
    img_np = np.array(img_pil.convert('RGB'))
    h_img, w_img = img_np.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(w_img, int(x2 + bw * padding))
    y2 = min(h_img, int(y2 + bh * padding))
    crop = img_np[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    h, w = enhanced.shape
    short_side = min(h, w)
    scale = max(2, 300 // max(short_side, 1))
    enhanced = cv2.resize(enhanced, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'debug_crops')
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f'crop_{idx}.png'), enhanced)
    print(f"[CROP DEBUG] crop_{idx} 크기: {enhanced.shape}")

    return enhanced


def read_imprint(img_np):
    reader = get_ocr_reader()
    best_text = ''
    best_conf = 0.0
    for k in range(4):
        rotated = np.rot90(img_np, k=k)
        results = reader.readtext(
            rotated, detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            contrast_ths=0.05, adjust_contrast=0.7,
            text_threshold=0.3, low_text=0.3
        )
        if not results:
            continue
        combined_text = re.sub(r'[^A-Z0-9]', '', ''.join([r[1] for r in results]).upper())
        avg_conf = sum(r[2] for r in results) / len(results)
        print(f"[OCR DEBUG] 회전 {k * 90}도 → 텍스트: '{combined_text}', 신뢰도: {avg_conf:.2f}")

        if combined_text and avg_conf > best_conf:
            best_conf = avg_conf
            best_text = combined_text

    MIN_OCR_CONFIDENCE = 0.5
    if best_conf < MIN_OCR_CONFIDENCE:
        print(f"[OCR DEBUG] 최종 신뢰도({best_conf:.2f})가 기준치 미달 — 인식 실패로 처리")
        return ''

    return best_text


def lookup_pill_by_imprint(imprint_text):
    if not imprint_text:
        return None
    conn = sqlite3.connect(PILL_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('SELECT * FROM pill_identity WHERE print_front = ? LIMIT 1', (imprint_text,))
    row = cur.fetchone()
    if not row:
        cur.execute('SELECT * FROM pill_identity WHERE print_front LIKE ? LIMIT 1', (f'%{imprint_text}%',))
        row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


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
    import requests as http_requests

    if 'image' not in request.files:
        return jsonify({'error': 'Image file required'}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    yolo = load_yolo()
    results = yolo(img)

    detections = []
    boxes = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo.names[cls].upper()
            xyxy = box.xyxy[0].tolist()
            detections.append({'label': label, 'confidence': round(conf * 100, 1)})
            boxes.append(xyxy)

    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_img)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    drug_hint = request.form.get('drugname', '').strip()
    api_key = os.environ.get('MFDS_API_KEY', '')
    mfds_results = []
    searched = set()

    for idx, box in enumerate(boxes[:3]):
        ocr_text = ''
        try:
            crop = crop_and_preprocess(img, box, idx)
            ocr_text = read_imprint(crop)
            print(f"[OCR DEBUG] 인식된 텍스트: '{ocr_text}'")
        except Exception as e:
            print(f"OCR error: {str(e)}")

        search_key = drug_hint.upper().replace(' ', '') if drug_hint else ocr_text
        if not search_key or search_key in searched:
            continue
        searched.add(search_key)

        try:
            item_name = None
            drug_detail = None

            if not drug_hint:
                local_match = lookup_pill_by_imprint(ocr_text)
                print(f"[LOCAL DB DEBUG] 검색어: '{ocr_text}' → 매칭: {local_match}")
                if local_match:
                    item_name = local_match.get('item_name', '-')
                    drug_detail = {
                        'detected_imprint': ocr_text or '-',
                        'used_hint': '-',
                        'name': item_name,
                        'company': local_match.get('entp_name', '-'),
                        'shape': local_match.get('drug_shape', '-'),
                        'color': local_match.get('color_class1', '-'),
                        'img_url': local_match.get('item_image', ''),
                        'class_name': local_match.get('class_name', '-'),
                    }
            else:
                url = 'https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03'
                params = {
                    'serviceKey': api_key,
                    'item_name': drug_hint,
                    'pageNo': 1,
                    'numOfRows': 1,
                    'type': 'json'
                }
                res = http_requests.get(url, params=params, timeout=10)
                items = res.json().get('body', {}).get('items', [])
                if items:
                    item = items[0]
                    item_name = item.get('ITEM_NAME', '-')
                    drug_detail = {
                        'detected_imprint': ocr_text or '-',
                        'used_hint': drug_hint,
                        'name': item_name,
                        'company': item.get('ENTP_NAME', '-'),
                        'shape': item.get('DRUG_SHAPE', '-'),
                        'color': item.get('COLOR_CLASS1', '-'),
                        'img_url': item.get('ITEM_IMAGE', ''),
                        'class_name': item.get('CLASS_NAME', '-'),
                    }

            if drug_detail and item_name and item_name != '-':
                try:
                    dur_url = 'https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList'
                    dur_params = {
                        'serviceKey': api_key,
                        'itemName': item_name,
                        'pageNo': 1,
                        'numOfRows': 1,
                        'type': 'json'
                    }
                    dur_res = http_requests.get(dur_url, params=dur_params, timeout=10)
                    dur_items = dur_res.json().get('body', {}).get('items', [])
                    if dur_items:
                        d = dur_items[0]
                        drug_detail.update({
                            'efficacy': d.get('efcyQesitm', '-'),
                            'usage': d.get('useMethodQesitm', '-'),
                            'warning': d.get('atpnWarnQesitm', '-'),
                            'precaution': d.get('atpnQesitm', '-'),
                            'interaction': d.get('intrcQesitm', '-'),
                            'side_effect': d.get('seQesitm', '-'),
                            'storage': d.get('depositMethodQesitm', '-'),
                        })
                except Exception as e:
                    print(f"DUR API error: {str(e)}")

            if drug_detail:
                mfds_results.append(drug_detail)
        except Exception as e:
            print(f"MFDS lookup error: {str(e)}")

    return jsonify({
        'detections': detections,
        'image': img_b64,
        'mfds_results': mfds_results
    })
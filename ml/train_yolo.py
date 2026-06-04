from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')

results = model.train(
    data=os.path.abspath('ml/medical_pills.yaml'),
    epochs=50,
    imgsz=640,
    batch=8,
    name='pill_detector',
    device='cpu'
)

print("✅ 학습 완료!")
print("best.pt 위치: runs/detect/pill_detector/weights/best.pt")
import cv2
from ultralytics import YOLO

# Modeli yükle (.pt dosyasının yolu)
model = YOLO(r"C:\Users\ereno\runs\detect\train9\weights\best.pt")  # senin model yoluna göre güncelle

# Kamerayı başlat (0: varsayılan kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı.")
        break

    # Modeli kullanarak tahmin yap
    results = model(frame, stream=True)

    # Sonuçları çiz
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            
            # Koordinatları al ve çiz
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Okey Tasi Tespiti", frame)

    # ESC ile çık
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()

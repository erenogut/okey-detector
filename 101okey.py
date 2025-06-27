import cv2
from ultralytics import YOLO

# Load the trained YOLO model (change the path to your own .pt file)
model = YOLO(r"C:\Users\ereno\runs\detect\train9\weights\best.pt")  # senin model yoluna göre güncelle

# Start the webcam (0 = default camera)
cap = cv2.VideoCapture(0)
# Check if the camera is working
if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı.")
        break

    # Run YOLO model on the frame
    results = model(frame, stream=True)

    # Draw detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            
            # Get bounding box coordinates and draw it
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("Okey Tasi Tespiti", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

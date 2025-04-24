import cv2
import numpy as np
import pytesseract
from tensorflow.keras.models import load_model
from datetime import datetime
import csv
import os

# ========== CONFIGURATION ==========

model = load_model('model/emission_cnn.h5')
class_names = ['heavy_smoke', 'light_smoke', 'no_smoke']
img_size = (128, 128)

os.makedirs('logs', exist_ok=True)
os.makedirs('plates', exist_ok=True)

csv_path = 'logs/smoke_with_plate_log.csv'
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Class', 'Carbon Density', 'Plate Number', 'Plate Image'])

# ========== START REAL-TIME CAMERA ==========

cap = cv2.VideoCapture(0)
print("🚀 Real-time detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    resized = cv2.resize(frame, img_size)
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)

    pred = model.predict(input_data)[0]
    class_index = np.argmax(pred)
    confidence = pred[class_index]
    carbon_density = confidence * 100

    # Determine class label and display % based on carbon density
    if carbon_density <= 50:
        class_label = 'No Smoke'
        carbon_density_display = "25%"
        border_color = (0, 255, 0)  # Green
    elif 50 < carbon_density <= 70:
        class_label = 'Light Smoke'
        carbon_density_display = "60%"
        border_color = (0, 255, 255)  # Yellow
    else:
        class_label = 'Heavy Smoke'
        carbon_density_display = "85%"
        border_color = (0, 0, 255)  # Red

    # Display probabilities in terminal
    for i, prob in enumerate(pred):
        print(f"{class_names[i]}: {prob * 100:.2f}%")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plate_text = "N/A"
    plate_image_path = "N/A"

    if class_label != 'No Smoke':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                plate_img = frame[y:y + h, x:x + w]

                plate_text = pytesseract.image_to_string(plate_img, config='--psm 8 --oem 3')
                plate_text = plate_text.strip().replace('\n', '').replace(' ', '')

                plate_image_path = f'plates/{timestamp}_{plate_text}.jpg'
                cv2.imwrite(plate_image_path, plate_img)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                break

    # Draw a colored border around the frame
    frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), border_color, 10)

    # Draw black bar on top for clean text
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)

    # Compose final label
    overlay_text = f"{timestamp} | {class_label} | Carbon Density: {carbon_density_display} | Plate: {plate_text[:10]}"
    cv2.putText(frame, overlay_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)

    # Show the frame
    cv2.imshow("Vehicle Emission + Plate Detection", frame)

    # Save log to CSV
    csv_writer.writerow([
        timestamp,
        class_label,
        carbon_density_display,
        plate_text,
        plate_image_path
    ])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========== CLEANUP ==========

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"✅ Detection complete! Log saved to → {csv_path}")

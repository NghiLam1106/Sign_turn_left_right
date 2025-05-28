# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime
from PIL import Image
import serial
import time

# ==============================
# Load the trained model
# ==============================
model_path = r'E:\HOC_TAP\Nam_3\HOC_KY_2\Chuyen_de_2\source_code\traffic_classifier.h5'
model = load_model(model_path)

# ==============================
# Class labels
# ==============================
class_labels = ['Turn Right', 'Turn Left']

# ==============================
# Kết nối tới Arduino qua Serial
# ==============================
try:
    ser = serial.Serial('COM6', 9600, timeout=1)  # Thay COM5 bằng cổng đúng của bạn
    time.sleep(2)
    print("✅ Connected to Arduino.")
except Exception as e:
    print("❌ Error connecting to Arduino:", e)
    ser = None

# ==============================
# Current time function
# ==============================
def date_time():
    return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())

# ==============================
# IP camera
# ==============================
camera_url = "http://192.168.1.21:4747/video"
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("❌ Cannot open IP camera. Please check the URL or network connection.")
    exit()

frame_count = 0
skip_frame = 5
label = "Initializing..."

print("🚦 Starting prediction from camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received.")
        break

    frame = cv2.resize(frame, (640, 480))
    timestamp = date_time()

    frame_count += 1
    if frame_count % skip_frame == 0:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame).resize((128, 128))
            img_array = np.array(pil_img).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            if confidence >= 0.9:
                label = f"{class_labels[predicted_class]} ({confidence:.2f})"

                # Gửi tín hiệu đến Arduino nếu nhận diện tốt
                if ser:
                    if predicted_class == 0:  # Turn Right
                        ser.write(b'RIGHT\n')
                        print("📤 Sent command to Arduino: RIGHT")
                    elif predicted_class == 1:  # Turn Left
                        ser.write(b'LEFT\n')
                        print("📤 Sent command to Arduino: LEFT")
            else:
                label = "Unknown"

            print(f"[{timestamp}] 🔍 Prediction: {label}")

        except Exception as e:
            print("❌ Error processing image:", e)
            label = "Processing Error"

    # Hiển thị kết quả
    cv2.putText(frame, f'Prediction: {label}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, timestamp, (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Traffic Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹ Program exited.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()

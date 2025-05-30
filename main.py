import time
import random
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- CẤU HÌNH GPIO ---
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Chân điều khiển motor (BOARD numbering)
IN1 = 11  # Trái
IN2 = 13
IN3 = 15  # Phải
IN4 = 12
ENA = 16  # PWM trái
ENB = 18  # PWM phải

motor_pins = [IN1, IN2, IN3, IN4, ENA, ENB]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(100)
pwmB.start(100)

# --- ĐIỀU KHIỂN ---
def stop_all():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def turn_left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def turn_right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

# --- TRẠNG THÁI ---
current_mode = "idle"
last_action_time = 0
action_cooldown = 2  # giây

# --- TẢI MÔ HÌNH KERAS .h5 ---
model = load_model('traffic_classifier.h5')
class_names = ['Turn Right', 'Turn Left']

# --- CAMERA ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(config)
picam2.start()

# --- HÀNH ĐỘNG BIỂN BÁO ---
def execute_action(label):
    global current_mode, last_action_time
    now = time.time()

    if now - last_action_time < action_cooldown:
        return

    print(f"📸 Phát hiện biển báo: {label.upper()}")
    last_action_time = now

    # if label == "stop":
        # stop_all()
        # current_mode = "stopped"
    # if label == "straight":
    #     forward()
    #     current_mode = "forward"
    if label == "Turn Left":
        turn_left()
        time.sleep(1.0)
        stop_all()
        current_mode = "Turn Left"
    elif label == "Turn Right":
        turn_right()
        time.sleep(1.0)
        stop_all()
        current_mode = "Turn Right"

# --- TIỀN XỬ LÝ ẢNH ---
def preprocess(image):
    # Chuyển ảnh sang RGB nếu đang ở BGR (do OpenCV đọc ảnh ở BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize đúng kích thước mô hình huấn luyện
    image = cv2.resize(image, (128, 128))

    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    image = image.astype('float32') / 255.0

    # Thêm chiều batch để phù hợp với đầu vào của mô hình
    image = np.expand_dims(image, axis=0)

    return image

# --- VÒNG LẶP CHÍNH ---
try:
    while True:
        if current_mode == "idle":
            forward()
            current_mode = "forward"

        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        roi = frame_bgr[100:380, 150:490]  # Vùng nhận diện biển báo (cắt giữa ảnh)
        input_img = preprocess(roi)
        prediction = model.predict(input_img)
        predicted_index = np.argmax(prediction)
        label = class_names[predicted_index]

        confidence = prediction[0][predicted_index]
        if confidence > 0.8:
            execute_action(label)

        # Hiển thị
        cv2.putText(frame_bgr, f"{label} ({confidence*100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Robot Vision", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⛔ Dừng chương trình")

# --- DỌN DẸP ---
picam2.stop()
cv2.destroyAllWindows()
stop_all()
pwmA.stop()
pwmB.stop()
GPIO.cleanup()

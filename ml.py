import sys
import cv2
import face_recognition
import os
import numpy as np
import threading
import paho.mqtt.client as mqtt
import json
import time

# Home Assistant MQTT Broker
MQTT_BROKER = "homeassistant.local"  # Change to your HA IP or hostname
MQTT_PORT = 1883
MQTT_TOPIC = "homeassistant/face_recognition"

# Ensure UTF-8 encoding (for Windows)
sys.stdout.reconfigure(encoding='utf-8')

FOLDER_PATH = "pics"
RTSP_URL = "rtsp://admin:moataz2019@192.168.0.105:554/cam/realmonitor?channel=1&subtype=0"

# Load known face encodings
face_encodings_dict = {}

print("📥 Loading faces from images...")

for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(('.jpeg', '.jpg', '.png')):
        path = os.path.join(FOLDER_PATH, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            name = filename.split('.')[0].rstrip('0123456789')  # Remove numbers from filename
            face_encodings_dict.setdefault(name, []).append(encodings[0])

# Compute average encoding for each person
known_face_encodings = []
known_face_names = []

for name, encodings in face_encodings_dict.items():
    avg_encoding = np.mean(encodings, axis=0)  # Average multiple encodings
    known_face_encodings.append(avg_encoding)
    known_face_names.append(name)

print(f"✅ Loaded {len(known_face_names)} people for face recognition!")
print("🎥 Connecting to IP Camera...")

# Open IP Camera with low latency settings
video_capture = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not video_capture.isOpened():
    print("❌ ERROR: Could not connect to IP Camera! Check the RTSP URL.")
    exit(1)

# Reduce buffering and optimize FPS
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
video_capture.set(cv2.CAP_PROP_FPS, 30)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0  # Used to skip frames for better FPS
lock = threading.Lock()  # Lock for threading
current_frame = None  # Shared variable to store the latest frame

# MQTT Connection
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to Home Assistant MQTT Broker!")
    else:
        print(f"⚠ MQTT Connection failed with code {rc}")

mqtt_client = mqtt.Client("FaceRecognition")
mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

def send_mqtt_message(name, confidence):
    """Send recognized face data to Home Assistant via MQTT."""
    payload = json.dumps({"name": name, "confidence": confidence})
    mqtt_client.publish(MQTT_TOPIC, payload)
    print(f"📤 Sent MQTT: {payload}")

def read_frames():
    """Continuously reads frames from the IP camera to avoid delay."""
    global current_frame
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("⚠ WARNING: Unable to fetch frame. Reconnecting...")
            video_capture.open(RTSP_URL)  # Try to reconnect
            continue

        with lock:
            current_frame = frame  # Update the latest frame

# Start a separate thread for reading frames
threading.Thread(target=read_frames, daemon=True).start()

while True:
    with lock:
        if current_frame is None:
            continue  # Skip if no frame is available
        frame = current_frame.copy()

    frame_counter += 1
    if frame_counter % 3 != 0:  # Skip every 2 out of 3 frames for faster processing
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Reduce image size for faster processing
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.4, fy=0.4)

    # Use the faster HOG model (faster than CNN)
    face_locations = face_recognition.face_locations(small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations, num_jitters=1)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

        name = "Unknown"
        confidence = 0

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = (1 - face_distances[best_match_index]) * 100

        send_mqtt_message(name, confidence)  # Send detected face to Home Assistant

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
mqtt_client.disconnect()

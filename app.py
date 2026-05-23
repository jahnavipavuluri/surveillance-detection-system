from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import pygame
import os
import time

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize pygame for sound alarm
pygame.mixer.init()

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Global flag for alarm
alarm_playing = False
alarm_lock = threading.Lock()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ABNORMAL_CLASSES = ['person']  # YOLOv8 detects persons; motion analysis determines abnormality


def play_alarm():
    """Play alarm sound when abnormal activity is detected."""
    global alarm_playing
    with alarm_lock:
        if not alarm_playing:
            alarm_playing = True
            try:
                pygame.mixer.music.load('static/alarm.mp3')
                pygame.mixer.music.play()
                time.sleep(3)
            except Exception:
                pass
            alarm_playing = False


def is_abnormal_motion(fg_mask):
    """
    Detect abnormal motion using background subtraction.
    Returns True if sudden/large motion is detected.
    """
    motion_pixels = cv2.countNonZero(fg_mask)
    frame_area = fg_mask.shape[0] * fg_mask.shape[1]
    motion_ratio = motion_pixels / frame_area
    return motion_ratio > 0.08  # More than 8% of frame in motion = abnormal


def process_frame(frame):
    """
    Process a single frame:
    - Apply background subtraction
    - Run YOLOv8 detection
    - Draw bounding boxes and labels
    - Trigger alarm if abnormal
    """
    global alarm_playing

    fg_mask = bg_subtractor.apply(frame)
    abnormal = is_abnormal_motion(fg_mask)

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in ABNORMAL_CLASSES and abnormal:
            color = (0, 0, 255)  # Red for abnormal
            display_label = f"ABNORMAL: {label.upper()} {conf:.2f}"
            alarm_thread = threading.Thread(target=play_alarm, daemon=True)
            alarm_thread.start()
        else:
            color = (0, 255, 0)  # Green for normal
            display_label = f"{label.upper()} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, display_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def generate_live_frames():
    """Generator for live camera feed."""
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


def generate_video_frames(video_path):
    """Generator for uploaded video feed."""
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


# ── ROUTES ──

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/concept')
def concept():
    return render_template('concept.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        file = request.files.get('video')
        if file and file.filename != '':
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
            return render_template('detect.html', video_uploaded=True, video_path=file.filename)
    return render_template('detect.html', video_uploaded=False)


@app.route('/video_feed/live')
def video_feed_live():
    return Response(generate_live_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/uploaded/<filename>')
def video_feed_uploaded(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_video_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import os
import base64
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import cv2
import numpy as np

# Flask Initialization
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(minutes=30)

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)

# Indian Standard Time (IST)
IST = ZoneInfo("Asia/Kolkata")

# Path to ONNX Deep Learning Models
YUNET_MODEL = os.path.join(os.getcwd(), 'face_detection_yunet_2023mar.onnx')
SFACE_MODEL = os.path.join(os.getcwd(), 'face_recognition_sface_2021dec.onnx')

# Automatic fallback download if run outside Docker
if not os.path.exists(YUNET_MODEL):
    print("[DOWNLOADING] YuNet face detection model...")
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        YUNET_MODEL
    )

if not os.path.exists(SFACE_MODEL):
    print("[DOWNLOADING] SFace face recognition model...")
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        SFACE_MODEL
    )

# Initialize OpenCV DNN models
detector = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL,
    config='',
    input_size=(320, 320),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000
)

recognizer = cv2.FaceRecognizerSF.create(
    model=SFACE_MODEL,
    config=''
)

# Load and Pre-compute Features for Known Persons
KNOWN_FACES_DIR = os.path.join(os.getcwd(), 'Images_Attendance')
known_features = []
class_names = []

def extract_feature_from_image(img_bgr):
    h, w, _ = img_bgr.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_bgr)
    if faces is not None and len(faces) > 0:
        aligned_face = recognizer.alignCrop(img_bgr, faces[0])
        feature = recognizer.feature(aligned_face)
        return feature
    return None

if os.path.exists(KNOWN_FACES_DIR):
    for img_name in os.listdir(KNOWN_FACES_DIR):
        img_path = os.path.join(KNOWN_FACES_DIR, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            feat = extract_feature_from_image(img)
            if feat is not None:
                known_features.append(feat)
                class_names.append(os.path.splitext(img_name)[0].upper())
                print(f"[LOADED] Feature vector mapped for: {os.path.splitext(img_name)[0].upper()}")
            else:
                print(f"[WARNING] No face detected in: {img_name}")

# Attendance Logging Utilities
def get_attendance_filename():
    folder_path = os.path.join('static', 'Attendance Logs')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    today_date = datetime.now(IST).strftime('%d-%m-%Y')
    return os.path.join(folder_path, f'Attendance_{today_date}.csv')

def is_already_registered_this_hour(name):
    file_name = get_attendance_filename()
    if not os.path.exists(file_name):
        return False

    with open(file_name, 'r') as f:
        data_list = f.readlines()
        today_date = datetime.now(IST).strftime('%d/%m/%Y')
        current_hour = datetime.now(IST).strftime('%H')

        for line in data_list:
            parts = line.strip().split(',')
            if len(parts) == 3:
                entry_name, entry_time, entry_date = parts
                entry_hour = entry_time.split(':')[0]
                if entry_name == name and entry_date == today_date and entry_hour == current_hour:
                    return True
    return False

def mark_attendance(name):
    file_name = get_attendance_filename()
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write('Name,Time,Date\n')

    if is_already_registered_this_hour(name):
        return False

    with open(file_name, 'a') as f:
        time_now = datetime.now(IST)
        t_string = time_now.strftime('%H:%M:%S')
        d_string = time_now.strftime('%d/%m/%Y')
        f.writelines(f'{name},{t_string},{d_string}\n')
        return True

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)

def is_logged_in():
    return 'username' in session

# Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate():
    if request.method == 'POST':
        role = request.form.get('role')
        username = request.form.get('username')
        password = request.form.get('password')
        if not role or not username or not password:
            flash("All fields are required. Please fill out the form completely.", "danger")
            return redirect(url_for('authenticate'))
        user = User.query.filter_by(username=username, role=role).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['username'] = username
            session['role'] = role
            session.permanent = True
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials or role. Please try again.", "danger")
    return render_template('authenticate.html')

@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        flash("Please log in first.", "warning")
        return redirect(url_for('authenticate'))
    folder = os.path.join('static', 'Attendance Logs')
    log_files = [f for f in os.listdir(folder) if f.endswith('.csv')] if os.path.exists(folder) else []
    return render_template('dashboard.html', logs=log_files)

@app.route('/view_log/<log_file>')
def view_log(log_file):
    log_path = os.path.join('static', 'Attendance Logs', log_file)
    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            log_content = file.readlines()
        return render_template('logviewer.html', log_file=log_file, log_content=log_content)
    else:
        flash("File not found.", "danger")
        return redirect(url_for('dashboard'))

@app.route('/process_frame', methods=['POST'])
def process_frame():
    now_ist = datetime.now(IST)
    if now_ist.hour < 9 or now_ist.hour > 17:
        return jsonify({'status': 'error', 'message': 'Attendance allowed only between 9:00 AM and 5:00 PM IST.'})

    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'No image data'}), 400

    display_width = data.get('displayWidth', 480)
    display_height = data.get('displayHeight', 360)

    # Decode base64 frame
    encoded_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w, _ = img.shape
    scale_x = display_width / w
    scale_y = display_height / h

    # Detect faces via YuNet
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        return jsonify({
            'status': 'success',
            'name': 'No person detected',
            'detections': [],
            'marked': False
        })

    detections = []
    recognized_person = None
    marked = False

    for face in faces:
        fx, fy, fw, fh = face[0:4].astype(int)

        # Align face and compute 128-d deep embedding
        aligned_face = recognizer.alignCrop(img, face)
        live_feature = recognizer.feature(aligned_face)

        name = "UNKNOWN"
        best_score = -1.0
        best_idx = -1

        # Match against known feature vectors via Cosine Similarity
        for idx, k_feat in enumerate(known_features):
            score = recognizer.match(k_feat, live_feature, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Standard SFace cosine similarity match threshold
        if best_idx != -1 and best_score >= 0.363:
            name = class_names[best_idx]
            recognized_person = name
            if not is_already_registered_this_hour(name):
                marked = mark_attendance(name)

        # Map bounding box back to browser display dimensions
        top = int(fy * scale_y)
        left = int(fx * scale_x)
        bottom = int((fy + fh) * scale_y)
        right = int((fx + fw) * scale_x)

        detections.append({
            'box': [top, right, bottom, left],
            'name': name
        })

    return jsonify({
        'status': 'success',
        'name': recognized_person if recognized_person else 'No person detected',
        'detections': detections,
        'marked': marked
    })

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('authenticate'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

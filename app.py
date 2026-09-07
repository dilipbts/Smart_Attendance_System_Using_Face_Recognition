import os
import io
import csv
import base64
import time
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import cv2
import numpy as np

# Flask Initialization
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_attendance_secret_key')

# Read PostgreSQL URL or fallback to SQLite
db_url = os.getenv('DATABASE_URL', 'sqlite:///attendance_users.db')
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(minutes=30)

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)

# Indian Standard Time (IST)
IST = ZoneInfo("Asia/Kolkata")

# Path to ONNX Models
YUNET_MODEL = os.path.join(os.getcwd(), 'face_detection_yunet_2023mar.onnx')
SFACE_MODEL = os.path.join(os.getcwd(), 'face_recognition_sface_2021dec.onnx')

if not os.path.exists(YUNET_MODEL):
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        YUNET_MODEL
    )

if not os.path.exists(SFACE_MODEL):
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        SFACE_MODEL
    )

# Models
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

# Active Verification Cache
active_trackers = defaultdict(dict)

def evaluate_3s_motion(samples):
    """
    Checks landmark micro-variance and natural tremor over the 3-second buffer.
    Photos, prints, and phone screens maintain near-zero variance.
    """
    if len(samples) < 5:
        return False

    arr = np.array(samples)  # Shape: (N, 10)
    variance_sum = np.sum(np.var(arr, axis=0))

    # Threshold for real human micro-movements vs flat rigid displays
    return bool(variance_sum > 0.00014)

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)

class AttendanceRecord(db.Model):
    __tablename__ = 'attendance_records'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(50), default="PRESENT")

class SpoofRecord(db.Model):
    __tablename__ = 'spoof_records'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(50), default="FRAUD_ATTEMPT_DETECTED")

# Load Known Faces
KNOWN_FACES_DIR = os.path.join(os.getcwd(), 'Images_Attendance')
known_features = []
class_names = []

def extract_feature_from_image(img_bgr):
    h, w, _ = img_bgr.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_bgr)
    if faces is not None and len(faces) > 0:
        aligned_face = recognizer.alignCrop(img_bgr, faces[0])
        return recognizer.feature(aligned_face)
    return None

if os.path.exists(KNOWN_FACES_DIR):
    for img_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        img_path = os.path.join(KNOWN_FACES_DIR, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            feat = extract_feature_from_image(img)
            if feat is not None:
                known_features.append(feat)
                class_names.append(os.path.splitext(img_name)[0].upper())

# Database Logging Helpers
def mark_attendance(name):
    now = datetime.now(IST)
    d_string = now.strftime('%d/%m/%Y')
    current_hour = now.strftime('%H')

    recent = AttendanceRecord.query.filter_by(name=name, date=d_string).all()
    for rec in recent:
        if rec.time.split(':')[0] == current_hour:
            return False

    new_record = AttendanceRecord(
        name=name,
        time=now.strftime('%H:%M:%S'),
        date=d_string,
        status="PRESENT"
    )
    db.session.add(new_record)
    db.session.commit()
    return True

def record_spoof_attempt(name):
    now = datetime.now(IST)
    d_string = now.strftime('%d/%m/%Y')
    current_hour = now.strftime('%H')

    recent = SpoofRecord.query.filter_by(name=name, date=d_string).all()
    for rec in recent:
        if rec.time.split(':')[0] == current_hour:
            return False

    new_spoof = SpoofRecord(
        name=name,
        time=now.strftime('%H:%M:%S'),
        date=d_string,
        status="FRAUD_ATTEMPT_DETECTED"
    )
    db.session.add(new_spoof)
    db.session.commit()
    return True

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
            flash("All fields are required.", "danger")
            return redirect(url_for('authenticate'))

        user = User.query.filter(
            User.username == username,
            db.func.lower(User.role) == role.lower()
        ).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['username'] = user.username
            session['role'] = user.role
            session.permanent = True
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials or role.", "danger")
    return render_template('authenticate.html')

@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        flash("Please log in first.", "warning")
        return redirect(url_for('authenticate'))

    att_dates = [r[0] for r in db.session.query(AttendanceRecord.date).distinct().all()]
    spf_dates = [r[0] for r in db.session.query(SpoofRecord.date).distinct().all()]

    logs = []
    for d in sorted(set(att_dates), reverse=True):
        logs.append(f"Attendance_{d.replace('/', '-')}.csv")
    for d in sorted(set(spf_dates), reverse=True):
        logs.append(f"Spoof_Logs_{d.replace('/', '-')}.csv")

    return render_template('dashboard.html', logs=logs)

@app.route('/view_log/<log_file>')
def view_log(log_file):
    date_str = log_file.replace('Attendance_', '').replace('Spoof_Logs_', '').replace('.csv', '').replace('-', '/')
    log_content = ["Name,Time,Date,Status\n"]
    if "Spoof" in log_file:
        records = SpoofRecord.query.filter_by(date=date_str).order_by(SpoofRecord.id.desc()).all()
    else:
        records = AttendanceRecord.query.filter_by(date=date_str).order_by(AttendanceRecord.id.desc()).all()

    for r in records:
        log_content.append(f"{r.name},{r.time},{r.date},{r.status}\n")

    return render_template('logviewer.html', log_file=log_file, log_content=log_content)

@app.route('/download_log/<log_file>')
def download_log(log_file):
    date_str = log_file.replace('Attendance_', '').replace('Spoof_Logs_', '').replace('.csv', '').replace('-', '/')
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Time", "Date", "Status"])

    if "Spoof" in log_file:
        records = SpoofRecord.query.filter_by(date=date_str).all()
    else:
        records = AttendanceRecord.query.filter_by(date=date_str).all()

    for r in records:
        writer.writerow([r.name, r.time, r.date, r.status])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={log_file}"}
    )

@app.route('/process_frame', methods=['POST'])
def process_frame():
    now_ist = datetime.now(IST)
    if now_ist.hour == 23 and now_ist.minute > 50:
        return jsonify({'status': 'error', 'message': 'Attendance portal closed between 11:50 PM and 12:00 AM IST.'})

    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'No image data'}), 400

    display_width = data.get('displayWidth', 480)
    display_height = data.get('displayHeight', 360)

    encoded_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w, _ = img.shape
    scale_x = display_width / w
    scale_y = display_height / h

    detector.setInputSize((w, h))
    _, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        return jsonify({
            'status': 'success',
            'name': 'No person detected',
            'detections': [],
            'action_status': ''
        })

    detections = []
    action_status = ''
    primary_name = 'No person detected'
    current_time = time.time()

    for face in faces:
        # 1. Identity Recognition
        aligned_face = recognizer.alignCrop(img, face)
        live_feature = recognizer.feature(aligned_face)

        person_name = "UNKNOWN"
        best_score = -1.0
        best_idx = -1

        for idx, k_feat in enumerate(known_features):
            score = recognizer.match(k_feat, live_feature, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx != -1 and best_score >= 0.363:
            person_name = class_names[best_idx]

        primary_name = person_name

        # 2. Normalized 5-point facial landmarks calculation (Broadcasting Fix)
        fx, fy, fw, fh = face[0:4].astype(int)
        raw_landmarks = face[4:14].reshape((5, 2)).astype(float)
        norm_landmarks = ((raw_landmarks - [fx, fy]) / [max(fw, 1), max(fh, 1)]).flatten()

        # 3. 3-Second Temporal Verification Buffer
        tracker = active_trackers[person_name]

        if "last_seen" in tracker and (current_time - tracker["last_seen"]) > 2.5:
            tracker.clear()

        if "start_time" not in tracker:
            tracker["start_time"] = current_time
            tracker["samples"] = []
            tracker["triggered"] = False
            tracker["status"] = "ANALYZING"

        tracker["last_seen"] = current_time
        tracker["samples"].append(norm_landmarks)

        elapsed = current_time - tracker["start_time"]

        # Phase 1: Under 3.0 seconds -> Do not write to database
        if elapsed < 3.0 and not tracker["triggered"]:
            remaining = max(1, 3 - int(elapsed))
            display_tag = f"{person_name} (Verifying: {remaining}s)"
            box_color = '#FFD700'
            action_status = f"Face detected: Verifying liveness ({remaining}s remaining)..."

        # Phase 2: At or after 3.0 seconds -> Trigger database write once
        else:
            if not tracker["triggered"]:
                is_live = evaluate_3s_motion(tracker["samples"])
                tracker["triggered"] = True

                if is_live:
                    tracker["status"] = "LIVE"
                    if person_name != "UNKNOWN":
                        marked = mark_attendance(person_name)
                        action_status = f"VERIFIED: Attendance recorded for {person_name}!" if marked else f"{person_name} already logged this hour."
                    else:
                        action_status = "Live person confirmed, but identity is UNKNOWN."
                else:
                    tracker["status"] = "SPOOF"
                    record_spoof_attempt(person_name)
                    action_status = f"FRAUD DETECTED: Spoof attempt recorded for {person_name}!"

            if tracker["status"] == "LIVE":
                display_tag = f"LIVE: {person_name}"
                box_color = '#00FF00'
            else:
                display_tag = f"FRAUD / SPOOF: {person_name}"
                box_color = '#FF0000'

        top = int(fy * scale_y)
        left = int(fx * scale_x)
        bottom = int((fy + fh) * scale_y)
        right = int((fx + fw) * scale_x)

        detections.append({
            'box': [top, right, bottom, left],
            'name': display_tag,
            'color': box_color
        })

    return jsonify({
        'status': 'success',
        'name': primary_name,
        'detections': detections,
        'action_status': action_status
    })

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('authenticate'))

# Seed Default Users
with app.app_context():
    db.create_all()

    teacher = User.query.filter_by(username='Guru').first()
    if not teacher:
        hashed_teacher_pw = bcrypt.generate_password_hash('1234').decode('utf-8')
        db.session.add(User(username='Guru', password=hashed_teacher_pw, role='Teacher'))
    else:
        teacher.password = bcrypt.generate_password_hash('1234').decode('utf-8')
        teacher.role = 'Teacher'

    student = User.query.filter_by(username='Dilip DK').first()
    if not student:
        hashed_student_pw = bcrypt.generate_password_hash('demonking').decode('utf-8')
        db.session.add(User(username='Dilip DK', password=hashed_student_pw, role='Student'))
    else:
        student.password = bcrypt.generate_password_hash('demonking').decode('utf-8')
        student.role = 'Student'

    db.session.commit()

if __name__ == '__main__':
    app.run(debug=True)

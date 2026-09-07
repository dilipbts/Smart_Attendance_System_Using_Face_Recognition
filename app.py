import os
import base64
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import cv2
import numpy as np
import face_recognition

# Flask Initialization
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(minutes=30)

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)

# Timezone configuration (IST)
IST = ZoneInfo("Asia/Kolkata")

# Load and encode student/known faces
path = os.path.join(os.getcwd(), 'Images_Attendance')
images = []
class_names = []

if os.path.exists(path):
    image_list = os.listdir(path)
    for img_name in image_list:
        img = cv2.imread(os.path.join(path, img_name))
        if img is not None:
            images.append(img)
            class_names.append(os.path.splitext(img_name)[0])

def find_encodings(imgs):
    encode_list = []
    for img in imgs:
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodes = face_recognition.face_encodings(rgb_img)
            if len(encodes) > 0:
                encode_list.append(encodes[0])
        except Exception as e:
            print(f"[WARNING] Encoding failed: {e}")
    return encode_list

encode_list_known = find_encodings(images)

# Attendance File Utilities
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
        return jsonify({'status': 'error', 'message': 'Attendance is only allowed between 9:00 AM and 5:00 PM IST.'})

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'No image data'}), 400

    # Decode base64 frame
    encoded_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Downscale frame for quick detection
    img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(img_rgb)
    encodings = face_recognition.face_encodings(img_rgb, faces)

    detections = []
    recognized_person = None
    marked = False

    for encode_face, face_loc in zip(encodings, faces):
        name = "UNKNOWN"
        if len(encode_list_known) > 0:
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_distances = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = np.argmin(face_distances)

            if matches[match_index]:
                name = class_names[match_index].upper()
                recognized_person = name
                if not is_already_registered_this_hour(name):
                    marked = mark_attendance(name)

        # Scale coordinates back up to video element size (* 4)
        top, right, bottom, left = [v * 4 for v in face_loc]
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

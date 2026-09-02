from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta

# Flask Initialization
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(minutes=30)

# Extensions Initialization
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)

# Path to the folder containing images
path = os.path.join(os.getcwd(), 'Images_Attendance')
images = []
class_names = []
image_list = os.listdir(path)

# Read and store images and class names
for img_name in image_list:
    img = cv2.imread(os.path.join(path, img_name))
    if img is not None:
        images.append(img)
        class_names.append(os.path.splitext(img_name)[0])

# Encode faces
def find_encodings(images):
    encode_list = []
    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encode_list.append(encode)
        except IndexError:
            print(f"[WARNING] No face encodings found in image.")
    return encode_list

encode_list_known = find_encodings(images)

# Global variables for webcam and attendance system
webcam = None
is_running = False
recognized_name = ""

# Utility Functions
def get_attendance_filename():
    folder_path = os.path.join('static', 'Attendance Logs')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    today_date = datetime.now().strftime('%d-%m-%Y')
    return os.path.join(folder_path, f'Attendance_{today_date}.csv')

def is_already_registered_this_hour(name):
    file_name = get_attendance_filename()
    if not os.path.exists(file_name):
        return False

    with open(file_name, 'r') as f:
        data_list = f.readlines()
        today_date = datetime.now().strftime('%d/%m/%Y')
        current_hour = datetime.now().strftime('%H')

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
        print(f"{name} already registered in this hour, skipping.")
        return False

    with open(file_name, 'a') as f:
        time_now = datetime.now()
        t_string = time_now.strftime('%H:%M:%S')
        d_string = datetime.now().strftime('%d/%m/%Y')
        f.writelines(f'{name},{t_string},{d_string}\n')
        print(f"{name} has been registered.")
        return True

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)

# Flask Routes
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
    log_files = [f for f in os.listdir(os.path.join('static', 'Attendance Logs')) if f.endswith('.csv')]
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

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global is_running, webcam
    if not is_running:
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            return {"message": "Failed to start webcam"}
        is_running = True
        return {"message": "Webcam started successfully!"}
    return {"message": "Webcam is already running!"}

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global is_running, webcam
    if is_running:
        is_running = False
        if webcam:
            webcam.release()
        return {"message": "Webcam stopped successfully!"}
    return {"message": "Webcam is not running!"}

@app.route('/webcam_feed')
def webcam_feed():
    def generate_frames():
        global is_running, recognized_name

        # Restrict operation to 9 AM - 5 PM
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour > 17:
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n'
                   b'Attendance recognition is allowed only between 9 AM and 5 PM.\r\n')
            return

        while is_running:
            success, img = webcam.read()
            if not success:
                break
            img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(img_rgb)
            encodings = face_recognition.face_encodings(img_rgb, faces)
            for encode_face, face_loc in zip(encodings, faces):
                matches = face_recognition.compare_faces(encode_list_known, encode_face)
                face_distances = face_recognition.face_distance(encode_list_known, encode_face)
                match_index = np.argmin(face_distances)
                if matches[match_index]:
                    recognized_name = class_names[match_index].upper()
                    y1, x2, y2, x1 = [v * 4 for v in face_loc]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, recognized_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    if not is_already_registered_this_hour(recognized_name):
                        mark_attendance(recognized_name)
            _, buffer = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('authenticate'))

@app.route('/get_recognized_name', methods=['GET'])
def get_recognized_name():
    global recognized_name
    return {"name": recognized_name}

def is_logged_in():
    return 'username' in session

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

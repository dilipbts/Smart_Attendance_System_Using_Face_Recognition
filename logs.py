import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the folder containing images
path = 'Images_Attendance'
images = []
class_names = []
image_list = os.listdir(path)
print(image_list)

# Read and store images and class names
for img_name in image_list:
    img = cv2.imread(f'{path}/{img_name}')
    images.append(img)
    class_names.append(os.path.splitext(img_name)[0])
print(class_names)


def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def get_attendance_filename():
    # Path to the Attendance Logs folder
    folder_path = 'static\Attendance Logs'

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate the file name based on the current date
    today_date = datetime.now().strftime('%d-%m-%Y')
    return f'{folder_path}/Attendance_{today_date}.csv'


def is_already_registered_this_hour(name):
    file_name = get_attendance_filename()  # Get today's file
    if not os.path.exists(file_name):
        return False  # If file doesn't exist, no records yet

    with open(file_name, 'r') as f:
        data_list = f.readlines()
        today_date = datetime.now().strftime('%d/%m/%Y')
        current_hour = datetime.now().strftime('%H')  # Get current hour as a string

        for line in data_list:
            parts = line.strip().split(',')
            if len(parts) == 3:
                entry_name, entry_time, entry_date = parts
                entry_hour = entry_time.split(':')[0]  # Extract the hour from the time
                if entry_name == name and entry_date == today_date and entry_hour == current_hour:
                    return True
    return False


def mark_attendance(name):
    file_name = get_attendance_filename()  # Get today's file

    # Create the file if it doesn't exist
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write('Name,Time,Date\n')  # Write the header

    if is_already_registered_this_hour(name):
        print(f"{name} already registered in this hour, skipping.")
        return False

    with open(file_name, 'a') as f:
        time_now = datetime.now()
        t_string = time_now.strftime('%H:%M:%S')
        d_string = time_now.strftime('%d/%m/%Y')
        f.writelines(f'{name},{t_string},{d_string}\n')
        print(f"{name} has been registered.")
        return True


try:
    encode_list_known = find_encodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    start_time = datetime.strptime("09:00:00", '%H:%M:%S').time()
    end_time = datetime.strptime("17:00:00", '%H:%M:%S').time()

    while True:
        current_time = datetime.now().time()
        if current_time < start_time or current_time > end_time:
            print("Attendance can only be recorded between 9 AM and 5 PM.")
            break

        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        faces_current_frame = face_recognition.face_locations(img_rgb)
        encodes_current_frame = face_recognition.face_encodings(img_rgb, faces_current_frame)

        for encode_face, face_loc in zip(encodes_current_frame, faces_current_frame):
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_distance = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = np.argmin(face_distance)

            if matches[match_index]:
                name = class_names[match_index].upper()
                y1, x2, y2, x1 = [v * 4 for v in face_loc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                if not is_already_registered_this_hour(name):
                    if mark_attendance(name):
                        print(f"{name} now registered.")
            else:
                print("Face not recognized")

        cv2.imshow('webcam', img)
        if cv2.waitKey(10) == 13:  # Press 'Enter' key to break
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pymongo import MongoClient
import time
from datetime import datetime
from win32com.client import Dispatch
import os
import csv

# Function to speak
def speak(message):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(message)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["attendance_system"]
students_collection = db["students"]
attendance_collection = db["attendance"]

# Load face data from MongoDB
students = list(students_collection.find())
faces = []
labels = []
usns = []

for student in students:
    name = student["name"]
    usn = student["usn"]
    face_data = np.array(student["faces_data"])  # Get the face data from MongoDB
    faces.append(face_data)
    labels.extend([name] * len(face_data))  # Label each face with the student's name
    usns.extend([usn] * len(face_data))  # Store the corresponding USNs

faces = np.concatenate(faces)  # Convert list of arrays to one large array
print("Shape of Faces matrix:", faces.shape)

# Train KNN classifier with the face data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, labels)

# Initialize webcam
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'USN', 'DATE', 'TIME']

# Main loop to capture and process frames
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_in_frame = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_in_frame:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Predict the label (student name) for the detected face
        name = knn.predict(resized_img)[0]
        person_index = labels.index(name)
        usn = usns[person_index]  # Get the USN for the detected person

        # Get timestamp for attendance
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        # Save attendance to MongoDB
        existing_attendance = attendance_collection.find_one({"name": name, "date": date})
        if not existing_attendance:
            attendance_collection.insert_one({
                "name": name,
                "usn": usn,
                "date": date,
                "time": timestamp
            })
            print(f"Attendance taken for {name} on {date} at {timestamp}")
            speak(f"Attendance taken for {name}")

        # Display bounding boxes and name on the frame
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, name, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance = [name, usn, date, timestamp] 
    # Show the frame
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1)
    
    # If 'o' is pressed, take attendance
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(2)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)


    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

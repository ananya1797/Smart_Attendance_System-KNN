import cv2
import numpy as np
from pymongo import MongoClient

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")  # Ensure MongoDB is running
db = client["attendance_system"]
students_collection = db["students"]

# Webcam setup and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

name = input("Enter Your Name: ")
usn = input("Enter Your USN: ")

faces_data = []
i = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(faces_data) <= 100 and i % 10 == 0:  # Capture face data every 10 frames
            faces_data.append(resized_img.flatten())  # Flatten the image data

        i += 1

        # Display face capture progress on the webcam feed
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Prepare and insert data into MongoDB
faces_data_np = np.array(faces_data).tolist()  # Convert to list for MongoDB compatibility

student_data = {
    "name": name,
    "usn": usn,
    "faces_data": faces_data_np  # Store the flattened face data
}

# Insert the student data into MongoDB
students_collection.insert_one(student_data)
print(f"Data for {name} (USN: {usn}) has been saved to MongoDB.")

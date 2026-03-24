import cv2
import os
import time
import sys
import json

# Unlock the camera automatically
os.system("sudo chmod 666 /dev/video*")

# Open camera 
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
time.sleep(2) 

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

user_id = input('\n Enter User ID (numeric only) ==>  ')
user_name = input(' Enter User Name ==>  ')

filename = 'data.json'
if os.path.exists(filename):
    with open(filename, 'r') as f:
        try:
            data_list = json.load(f)
        except json.JSONDecodeError:
            # Corrupt file case
            data_list = []
else:
    data_list = []
data_list.append(user_name)

with open(filename, 'w') as f:
    json.dump(data_list, f, indent=4)

# Create folder
user_folder = f"{user_id}_{user_name}"
path = os.path.join("dataset", user_folder)
if not os.path.exists(path):
    os.makedirs(path)

print(f"\n Starting capture for {user_name}.")

count = 0
while count < 100:
    ret, img = cam.read()
    if not ret or img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        # Save image inside the person specific folder
        file_name = f"User.{user_id}.{count}.jpg"
        cv2.imwrite(os.path.join(path, file_name), gray[y:y+h,x:x+w])
        
        print(f" Captured {count}/100 for {user_name}", end="\r")
        time.sleep(0.1)

print(f"\n Success! Saved 100 images in {path}")
cam.release()

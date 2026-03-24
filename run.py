import cv2
import time
import sys
import os
import json

recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read('trainer/trainer.yml')
except:
    print("Error: trainer.yml not found. Run train.py first!")
    sys.exit()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Importing list name made while collecting
filename = 'data.json'

if os.path.exists(filename):
    with open(filename, 'r') as f:
        names = json.load(f)
    print("Current list in JSON:")
    for index, item in enumerate(names):
        print(f"{index}: {item}")
entry_log = {}

cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
time.sleep(2)
print("\n Lab Monitor Active. (Press Ctrl+C to stop and see results)")

try:
    while True:
        ret, img = cam.read()
        if not ret or img is None: continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for(x,y,w,h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Lower confidence == better accuracy
            if (confidence < 75):
                name = names[id]
                if id not in entry_log:
                    entry_log[id] = time.time()
                    print(f" {name} entered the lab at {time.strftime('%H:%M:%S')}")
            else:
                # Saw an unidentifiable face
                pass

except KeyboardInterrupt:
    print("\n\n Stopping monitor...")

print("\n<<< FINAL LAB SESSION REPORT >>>")
if not entry_log:
    print(" No recognized people entered the lab.")
else:
    for uid, start in entry_log.items():
        total_seconds = time.time() - start
        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        user_name = names[uid] if uid < len(names) else f"User {uid}"
        print(f" - {user_name}: {mins} minutes and {secs} seconds")

cam.release()

import cv2
import numpy as np
from PIL import Image
import os

root_path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    faceSamples = []
    ids = []
    
    # Searching every dataset folder
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                try:
                    PIL_img = Image.open(image_path).convert('L')
                    img_numpy = np.array(PIL_img, 'uint8')
                    user_id = int(file.split(".")[1])
                    faces = detector.detectMultiScale(img_numpy)
                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y+h, x:x+w])
                        ids.append(user_id)
                        print(f" Adding face from ID {user_id}", end="\r")
                except Exception as e:
                    print(f" Error on {file}: {e}")
    return faceSamples, ids

print("\n Scanning all folders in dataset...")
faces, ids = getImagesAndLabels(root_path)

if len(faces) == 0:
    print("\n No faces found in any subfolders.")
else:
    print(f"\n Training on {len(faces)} images.")
    recognizer.train(faces, np.array(ids))
    
    if not os.path.exists('trainer'): os.makedirs('trainer')
    recognizer.write('trainer/trainer.yml') 
    print(" SUCCESS: trainer/trainer.yml generated.")

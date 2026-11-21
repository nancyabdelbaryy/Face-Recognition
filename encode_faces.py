
import face_recognition
import os
import json
import cv2

ENCODINGS_FILE = "registered_faces.json"
IMAGES_FOLDER = "images"

data = []

for image_name in os.listdir(IMAGES_FOLDER):
    if image_name.endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(IMAGES_FOLDER, image_name)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            name = os.path.splitext(image_name)[0]
            data.append({
                "name": name,
                "encoding": encodings[0].tolist()
            })
            print(f"✅ Registered: {name}")
        else:
            print(f"❌ No face found in {image_name}")

with open(ENCODINGS_FILE, "w") as f:
    json.dump(data, f, indent=2)

print("📁 All images processed.")

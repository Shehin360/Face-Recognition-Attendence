
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import concurrent.futures

path = 'Training_images'
classNames = []
attendance = set()

# Load images and class names
for cl in os.listdir(path):
    img = cv2.imread(os.path.join(path, cl))
    if img is not None:
        classNames.append(os.path.splitext(cl)[0])

# Find encodings for all images
def findEncodings(images):
    encodeList = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(rgb)
        if len(encode) > 0:
            encodeList.append(encode[0])
    return encodeList

images = [cv2.imread(os.path.join(path, cl)) for cl in os.listdir(path)]
with concurrent.futures.ThreadPoolExecutor() as executor:
    encodeListKnown = list(executor.map(findEncodings, [images]))[0]

# Start video capture
cap = cv2.VideoCapture(0)

# Set up face recognition
while True:
    ret, img = cap.read()
    if not ret:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodingsCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Use multi-threading to speed up face recognition
    with concurrent.futures.ThreadPoolExecutor() as executor:
        matches = list(executor.map(face_recognition.compare_faces, [encodeListKnown]*len(encodingsCurFrame), encodingsCurFrame))
        faceDis = list(executor.map(face_recognition.face_distance, [encodeListKnown]*len(encodingsCurFrame), encodingsCurFrame))

    # Identify faces and mark attendance
    for i, (match, faceDis) in enumerate(zip(matches, faceDis)):
        matchIndex = np.argmin(faceDis)
        if match[matchIndex]:
            name = classNames[matchIndex].upper()
            if name not in attendance:  # Skip marking present if already marked
                attendance.add(name)
                y1, x2, y2, x1 = facesCurFrame[i]
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # Save attendance to a file
                with open('Test101.csv', 'a') as f:
                    now = datetime.now().strftime("%I:%M:%S %p")
                    f.write(f'\n{name},{now}')

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

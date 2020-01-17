import cv2
import sys
import dlib
import numpy as np
import faceswap as fs

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
i = int(0)
f = int(0)

def faceswap():
    fs.faceswap()

def faceDetection(i,f):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces
    for (x,y,h,w) in faces:
        cv2.rectangle(frame, (x, y), (x+w+50, y+h+50), (238,130,238), 2) 
    cv2.imshow('Video', frame)
    if len(faces) > 0:
        i = i+1
        if f == 99:
            roi_color = frame[y:y + h+50, x:x + w+50] 
            print("[INFO] Object found. Saving locally.") 
            label = str('faces.jpg')
            cv2.imwrite(label, roi_color)
        return i
    else:
        return 0

while True:
    face = faceDetection(i,f)
    f = f + face
    print(f)
    # Display the resulting frame
    if f == 100:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
faceswap()
image = cv2.imread("test.png")
cv2.imshow("image", image)
cv2.waitKey(0)
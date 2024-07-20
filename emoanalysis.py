import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import cv2
from deepface import DeepFace

cap=cv2.VideoCapture(0)

face_classifier=cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile("/Users/krishna/Desktop/ai/haarcascade_frontalface_alt_tree.xml"))

while True:
    isTrue, frame=cap.read()
    height,width, _=frame.shape
    scale_percent=25
    new_w=int(width*scale_percent/100)
    new_h=int(height*scale_percent/100)

    resized=cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
    faces=face_classifier.detectMultiScale(resized)
    response=DeepFace.analyze(resized, actions=["emotion"], enforce_detection=False)
    
    a=response[0]["dominant_emotion"]
    print(a)
    for face in faces:
      x, y, h, w =face
      cv2.putText(resized, a, (x,y), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0), 1)
      new_frame=cv2.rectangle(resized, (x,y), (x+h,y+w), (0,255,0), thickness=1)
      resized_new=cv2.resize(resized,(width,height),interpolation=cv2.INTER_AREA)
      cv2.imshow("video",resized_new)
     
      print(a) 
      print(response)   
    if cv2.waitKey(30)==27:
        break
cap.release()
cv2.destroyAllWindows()

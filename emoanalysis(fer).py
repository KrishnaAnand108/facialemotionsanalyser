import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from fer import FER
import cv2

cap=cv2.VideoCapture(0)

face_classifier=cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile("/Users/krishna/Desktop/ai/haarcascade_frontalface_alt_tree.xml"))

while True:
    isTrue, frame=cap.read()
    
    
    faces=face_classifier.detectMultiScale(frame)
    emotion_detector=FER()
    result=emotion_detector.detect_emotions(frame)

    print(result)

    bounding_box=result[0]["box"]
    emotions=result[0]["emotions"]

    cv2.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),(0, 0, 255), 2,)
    emotion_name, score = emotion_detector.top_emotion(frame)
    for index, (emotion_name, score) in enumerate(emotions.items()):
        color = (211, 211,211) if score < 0.5 else (255, 0, 0)
        emotion_score = "{}:{}".format(emotion_name, "{:.2f}".format(score))
        cv2.putText(frame,emotion_score,
               (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),
               cv2.FONT_HERSHEY_COMPLEX,0.75,color,2)
       
    cv2.imshow("video",frame)
    if cv2.waitKey(30)==27:
        break
cap.release()
cv2.destroyAllWindows()

import os
from deepface import DeepFace
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("/Users/krishna/Desktop/ai/yolov8m-face.pt")
names = model.names
face_classifier=cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile("/Users/krishna/Desktop/ai/haarcascade_frontalface_alt_tree.xml"))

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

crop_dir_name = "ultralytics_crop"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)

video_writer = cv2.VideoWriter("object_cropping_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

idx = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)

    if boxes is not None:
        for box in boxes:
            idx += 1
            
            crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

            cv2.imwrite(os.path.join(crop_dir_name, str(idx) + ".png"), crop_obj)

            faces=face_classifier.detectMultiScale(crop_obj)
            response=DeepFace.analyze(crop_obj,actions=["emotion"],enforce_detection=False)
            a=response[0]["dominant_emotion"]
            annotator.box_label(box,label=a, color=(0,0,255),txt_color=(255,255,255))
            print(a)
    cv2.imshow("ultralytics", im0)
    video_writer.write(im0)
   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

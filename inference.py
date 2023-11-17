import cv2
import tensorflow as tf
import math
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import torch
import numpy as np


yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# inference
model = tf.keras.models.load_model("models/first_normal.h5")
save_vid = True

def inference(linkVideo): # This function is used to cycle through a video
    X = []
    idx = 0
    alert_count = 0
    video_path = linkVideo  # The path to the data file I use
    print(f"playing : {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    if save_vid:
        result_writer = cv2.VideoWriter('P3268-LV .avi', 
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)
    skipTime  = 0
    skipFrame = 0
    while True:    
        ret, frame = cap.read()
        skipTime = skipTime +1
        if not ret:
            break
        if 1==1:
#         if skipTime >= 30: # When skipTime has passed the first 30 frames, ie the first 1 second, proceed to Detect Person
            skipFrame = skipFrame +1 # The variable skipFrame means that every 5 frames I will detect 1 time, so in 1 second I will detect 6 times
            # print(skipFrame)
#             if  skipFrame  == 5: # When skipFrame = 5, I will detect person and assign skipFrame = 0 to run again
            if 1==1:
                skipFrame = 0
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False  
                result = yolo_model(frame)     # Detect Person
                frame.flags.writeable = True   
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                for (xmin, ymin, xmax,   ymax,  confidence,  clas) in result.xyxy[0].tolist(): # Loop through all the Persons present in the video, giving the x,y of each Person
                    c_lm = []
                    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
                            
#                             frame.flags.writeable = False  

                            resulta = pose.process(frame[int(ymin):int(ymax),int(xmin):int(xmax):])
#                             frame.flags.writeable = True   

                            if resulta.pose_landmarks and clas == 0: # class here is class, class = 0 means human
                                for (id, lm) in enumerate(resulta.pose_landmarks.landmark):
                                    if id > 10 and id not in [17,18,19,20,21,22] and id not in [29,30,31,32] :
                                        c_lm.append(lm.x)
                                        c_lm.append(lm.y)
#                                         c_lm.append(lm.z)
#                                         c_lm.append(lm.visibility)
                    if len(c_lm) > 0: # c_ lm used to save a person's x and y variables in a loop through each person, when saving, there will be a state that there is no x,y data to save 
                        X.append(c_lm) # with linkVideo being violent, we add data to X_violent
                
                    if len(X) >= idx+10:
                        X_inp = np.array(X[idx:idx+10])
                        pred = model.predict(X_inp.reshape(-1, 10, 24))
                        print(pred[0][0])
                        if pred[0][0] > 0.50:
                            # cv2.putText(frame, str(pred[0][0]), (10, 10), cv2.FONT_HERSHEY_COMPLEX, 2,(255,255,255),3)
                            alert_count += 1
                        else:
                            alert_count = 0
                        idx +=1
        if alert_count > 3:
            cv2.putText(frame, "Aggression Behaviour Detected!!!", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 2,(0,0,255),3)
        
        if save_vid:
            result_writer.write(frame)
        cv2.imshow("pose", frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()

    cap.release()
    if save_vid:
        result_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = "/Users/neo/work/violence_detection/aggressive-behavior/P3268-LV_20231017_132121.mp4"

    inference(video)
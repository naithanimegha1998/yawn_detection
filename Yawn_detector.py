import dlib
import cv2
import numpy as np
from time import sleep

WIDTH_IMG=480
HEIGHT_IMG=540
cap=cv2.VideoCapture(0)
cap.set(3,WIDTH_IMG)
cap.set(4,HEIGHT_IMG)
yawn=0

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class TooManyFaces (Exception):
    pass
class NoFaces(Exception):
    pass


def get_landmarks(im):
    rec = detector(im, 1)

    if len(rec) > 1:
        raise TooManyFaces
    if len(rec) == 0:
        raise NoFaces

    return np.array([[p.x, p.y] for p in predictor(im, rec[0]).parts()])

def annote_landmarks(im, landmarks):
    # Overlaying points on the image
    im=im.copy()
    #print(landmarks)
    #print(type(landmarks))
    for id, point in enumerate(landmarks):
        pos=(point[0],point[1])
        cv2.putText(im, str(id), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0,255,255))
    return im

def top_lip(landmarks):
    top_lip_pts=[]
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts=np.squeeze(np.array(top_lip_pts))
    top_lip_mean=np.mean(top_lip_all_pts, axis=0)
    #print(top_lip_mean)
    return int(top_lip_mean[1])


def bottom_lip(landmarks):
    bottom_lip_pts=[]
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts=np.squeeze(np.array(bottom_lip_pts))
    bottom_lip_mean=np.mean(bottom_lip_all_pts,axis=0)
    return int(bottom_lip_mean[1])

def mouth_open(img):
    landmarks=get_landmarks(img)
    if len(landmarks)==0:
        return img , 0
    img_with_landmarks=annote_landmarks(img,landmarks)
    top_lip_cntr=top_lip(landmarks)
    bottom_lip_cntr=bottom_lip(landmarks)
    lip_distance=abs(bottom_lip_cntr-top_lip_cntr)
    return img_with_landmarks, lip_distance

timer=8
while True:
    ret, img=cap.read()
    #cv2.imshow("Yawn Detection ", img)
    #sleep(3)
    img_landmarks, lip_dst=mouth_open(img)
    if lip_dst==0:
        continue
    if lip_dst>35:

        if timer>7:
            timer=0
            cv2.putText(img,"Subject is Yawning",(200,400),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            yawn+=1
            cv2.putText(img, "Yawn count : "+ str(yawn) , (200, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            #sleep(2)
            cv2.imshow("Live Landmarks ", img_landmarks)
            cv2.imshow("Yawn Detection ", img)
        else:
            timer+=1
            cv2.imshow("Live Landmarks ", img_landmarks)
            cv2.imshow("Yawn Detection ", img)
            continue
    else:
        cv2.imshow("Live Landmarks ", img_landmarks)
        cv2.imshow("Yawn Detection ",img)

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()
from facial_emotion_recognition import EmotionRecognition
import cv2

cam=cv2.VideoCapture(0)
er=EmotionRecognition(device='cpu')
while True:
    _,imgFr=cam.read()
    imgFr=er.recognise_emotion(imgFr,return_type='BGR')
    cv2.imshow("Emotion",imgFr)
    key=cv2.waitKey(97)
    if key == 97:
        break
cam.release()
cv2.destroyAllWindows()
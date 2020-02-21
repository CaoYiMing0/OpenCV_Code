

import cv2
videoFileName = r'E:\\testimage\\slow.mp4'
cap = cv2.VideoCapture(videoFileName)
fgbg = cv2.createBackgroundSubtractorMOG2()
thresh = 200

count = 0
while True:
    count+=1
    ret, frame = cap.read()
    if not ret:  # 没读取到当前帧，结束
        break
    fgmask = fgbg.apply(frame)
    bgImage = fgbg.getBackgroundImage()
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if count:
        cv2.imshow("dawd",fgmask)
        cv2.waitKey()


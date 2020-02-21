"""
Meanshift(均值偏移)
要在OpenCV中使用Meanshift，首先我们需要设置目标，找到其直方图，以便我们可以将目标反投影到每帧上以计算均值偏移。-
我们还需要提供窗口的初始位置。对于直方图，此处仅考虑色相。另外，为避免由于光线不足而产生错误的值，-
可以使用cv2.inRange（）函数丢弃光线不足的值。


视频效果不太好
"""
import numpy as np
import cv2
cap = cv2.VideoCapture('E:\\testimage\\slow.mp4')

# 获取视频的第一帧
ret,frame = cap.read()
# 设置窗口的初始位置
#r,h,c,w = 250,90,400,125  # simply hardcoded the values
c,r,w,h = 130,430,350,200  # simply hardcoded the values
track_window = (c,r,w,h)
# 设置ROI用于跟踪
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # 用meanshift获取新位置
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # 画在图像中
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break

cv2.destroyAllWindows()
cap.release()
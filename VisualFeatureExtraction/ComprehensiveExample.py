"""
基于背景提取的运动估计
没什么卵用，效果不太好
"""
import cv2

videoFileName = r'E:\\testimage\\slow.mp4'
cap = cv2.VideoCapture(videoFileName)
# 创建背景提取类对应的对象
fgbg = cv2.createBackgroundSubtractorMOG2()
thresh = 200

while True:
    ret, frame = cap.read()
    if not ret:  # 没读到当前帧，结束
        break
    #使用apply实时更新当前的背景图像，然后计算出前景对应的掩模图像
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 30, 0xff, cv2.THRESH_BINARY)
    #实时更新后的背景图像
    bgImage = fgbg.getBackgroundImage()
    #以下是OpenCV 4.x用法
    #3.x需要使用_,cnts，_
    _, cnts, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if (area < thresh):  # 区域面积小于指定阈值
            continue
        count += 1
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0xff, 0), 2)
    print('共检测到', count, '个目标', '\n')
    cv2.imshow("frame", frame)
    #cv2.imshow("Background", bgImage)

    key = cv2.waitKey(30)  # 每一帧间隔30ms
    if key == 27:  # 按下ESC键，退出
        break

cap.release()
cv2.destroyAllWindows()
"""
扩展：
    1.实现红虚线框区域检测
    2.实现轨迹分析（识别运动方向和路线）
    3.对前景目标进行分析（目标识别与分类）
    4.改善效果（减少阴影/目标粘连）
"""
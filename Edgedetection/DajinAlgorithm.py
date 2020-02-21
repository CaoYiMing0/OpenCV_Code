"""
大津算法计算阈值
cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst 第一个是阈值（T），第二个是处理过的图像
    src：表示的是图片源
    thresh：表示的是阈值（起始值）
    max你值
    type：表示的是这里划分的时候使用的是什么类型的算法，常用值为0（cv2.THRESH_BINARY）
"""

import cv2
import numpy as np

"""
img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 255  # create a image with black color in the middle and its background is white.
cv2.imshow("Image-Old", img)
ret, thresh = cv2.threshold(img, 127, 255, 0)
cv2.imshow("Image-New", thresh)
"""
filename = r'E:\testimage\red.jpg'
img = cv2.imread(filename, 0)
cv2.imshow("Image-Old", img)

ret, thresh = cv2.threshold(img, 200, 255, 0)  # 由灰度直方图确定最佳阈值
print(ret)
cv2.imshow("Image-New", thresh)

cv2.waitKey()
cv2.destroyAllWindows()

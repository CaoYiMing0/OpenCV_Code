"""
基于FlannBasedMatcher的SURF实现
SURF全称为“加速稳健特征”（Speeded Up Robust Feature）,不仅是尺度不变特征，而且是具有较高计算效率的特征。-
可被认为SURF是尺度不变特征变换算法（SIFT算法）的加速版。SURF最大的特征在于采用了haar特征以及积分图像的概念，-
SIFT采用的是DoG图像，而SURF采用的是Hessian矩阵（SURF算法核心）行列式近似值图像。SURF借鉴了SIFT算法中简化近似的思想，-
实验证明，SURF算法较SIFT算法在运算速度上要快3倍，综合性优于SIFT算法。

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = r'E:\testimage\lena.jpg'
imgname2 = r'E:\testimage\lena3.jpg'

surf = cv2.xfeatures2d.SURF_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

img1 = cv2.imread(imgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
kp1, des1 = surf.detectAndCompute(img1, None)  # des是描述子

img2 = cv2.imread(imgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = surf.detectAndCompute(img2, None)

img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])
img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imshow("SURF", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()

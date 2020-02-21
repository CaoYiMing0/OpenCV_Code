"""
SIFT (Scale-Invariant Feature Transform)  ————尺度不变特征变换

诸如Harris等的角检测器。它们是旋转不变的，这意味着即使图像旋转了，我们也可以找到相同的角。
但是缩放呢？如果缩放图像，则角可能不是角。

SIFT算法特点：
    图像的局部特征，对旋转、尺度缩放、亮度变化保持不变，对视角变化、仿射变换、噪声也保持一定程度的稳定性。
    独特性好，信息量丰富，适用于海量特征库进行快速、准确的匹配。
    多量性：即使是很少几个物体也可以产生大量的SIFT特征
    高速性：改进的SIFT匹配算法甚至可以达到实时性
    扩展性：可以很方便的与其他的特征向量进行联合。
SIFT的缺点:
    SIFT在图像的不变特征提取方面拥有无与伦比的优势，但并不完美，仍然存在：
    · 实时性差。
    · 有时特征点较少。
    · 对边缘光滑的目标无法准确提取特征点。
    等缺点，如下图所示，对模糊的图像和边缘平滑的图像，检测出的特征点过少，对圆更是无能为力。
    近来不断有人改进，其中最著名的有SURF；以及另辟蹊径的ORB等算子


基于FlannBasedMatcher的SIFT实现
FLANN(Fast_Library_for_Approximate_Nearest_Neighbors)快速最近邻搜索包，它是一个对大数据集和高维特征进行最近邻搜索的算法的集合,
而且这些算法都已经被优化过了。在面对大数据集时它的效果要好于 BFMatcher。
经验证，FLANN比其他的最近邻搜索软件快10倍。使用 FLANN 匹配,我们需要传入两个字典作为参数。这两个用来确定要使用的算法和其他相关参数等。
第一个是 IndexParams。
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) 。
这里使用的是KTreeIndex配置索引，指定待处理核密度树的数量（理想的数量在1-16）。
第二个字典是SearchParams。
search_params = dict(checks=100)用它来指定递归遍历的次数。值越高结果越准确，但是消耗的时间也越多。实际上，匹配效果很大程度上取-
决于输入。
5kd-trees和50checks总能取得合理精度，而且短时间完成。在之下的代码中，丢弃任何距离大于0.7的值，则可以避免几乎90%的错误匹配，但-
是好的匹配结果也会很少。

"""
import numpy as np
import cv2
import pandas as pd
from cv2 import flann
from matplotlib import pyplot as plt
from numpy.core._multiarray_umath import ndarray

imgname1 = r'E:\testimage\lena.jpg'
imgname2 = r'E:\testimage\lena3.jpg'

sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread(imgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子

img2 = cv2.imread(imgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
kp2, des2 = sift.detectAndCompute(img2, None)  # des是描述子

img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈

cv2.imshow("lena_point", img3)  # 拼接显示为gray
cv2.imshow("lena2_point", img4)  # 拼接显示为gray
# BFMatcher解决匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# 调整ratio
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # ratio=0. 4：对于准确度要求高的匹配； ratio=0. 6：对于匹配点数目要求比较多的匹配； ratio=0. 5：一般情况下。
        good.append([m])
# 基于FlannBasedMatcher的SIFT实现
img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
img6 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)  # 明显优于上面的匹配，并且为预想的匹配区域
cv2.imshow("FLANN", img5)
cv2.imshow("FLANN_good", img6)

cv2.waitKey()
cv2.destroyAllWindows()

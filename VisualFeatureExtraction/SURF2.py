import cv2
import numpy as np
imgname1 = r'E:\testimage\lena.jpg'
imgname2 = r'E:\testimage\lena3.jpg'
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
minHessian = 1000
detector = cv2.xfeatures2d.SIFT_create(minHessian)
descriptor = cv2.xfeatures2d.SIFT_create()
matcher1 = cv2.DescriptorMatcher_create("BruteForce")
# 检测特征点
keyPoint1 = descriptor.detect(img1)
keyPoint2 = descriptor.detect(img2)

#计算特征点对应描述子
_, descriptors1 = descriptor.compute(img1, keyPoint1)
_, descriptors2 = descriptor.compute(img2, keyPoint2)

# 描述子匹配
matches = matcher1.match(descriptors1, descriptors2)
img_matches = np.empty(img2.shape)
img_matches1 = cv2.drawMatches(img1, keyPoint1, img2, keyPoint2, matches, img_matches)
cv2.imshow("img_matches", img_matches1)
cv2.waitKey()
print("keyPoint1.size = ", len(keyPoint1))
print("keyPoint2.size = ", len(keyPoint2))

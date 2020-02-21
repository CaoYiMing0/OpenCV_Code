import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

'''
    opencv的boxFilter()函数和blur()函数都能用来进行均值平滑，其参数如下：
        cv2.boxFilter(src,ddepth,ksize,dst,anchor,normalize,borderType)
        src: 输入图像对象矩阵,
        ddepth:数据格式,位深度
        ksize:高斯卷积核的大小，格式为(宽，高)
        dst:输出图像矩阵,大小和数据类型都与src相同
        anchor：卷积核锚点，默认(-1,-1)表示卷积核的中心位置
        normalize:是否归一化 （若卷积核3*5，归一化卷积核需要除以15）
        borderType:填充边界类型
        
    cv2.blur(src,ksize,dst,anchor,borderType)
        src: 输入图像对象矩阵,可以为单通道或多通道
        ksize:高斯卷积核的大小，格式为(宽，高)
        dst:输出图像矩阵,大小和数据类型都与src相同
        anchor：卷积核锚点，默认(-1,-1)表示卷积核的中心位置
        borderType:填充边界类型   
'''
filename = r'E:\testimage\lena.jpg'
img = cv.imread(filename)
img_blur = cv.blur(img, (3, 5))
# img_blur = cv.boxFilter(img,-1,(3,5))
cv.imshow("img", img)
cv.imshow("img_blur", img_blur)
print(img.shape)
cv.waitKey()
cv.destroyAllWindows()

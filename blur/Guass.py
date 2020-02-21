import cv2 as cv

"""
    dst = cv2.GaussianBlur(src,ksize,sigmaX,sigmay,borderType)
        src: 输入图像矩阵,可为单通道或多通道，多通道时分别对每个通道进行卷积
        dst:输出图像矩阵,大小和数·据类型都与src相同
        ksize:高斯卷积核的大小，宽，高都为奇数，且可以不相同
        sigmaX: 一维水平方向高斯卷积核的标准差
        sigmaY: 一维垂直方向高斯卷积核的标准差，默认值为0，表示与sigmaX相同
        borderType:填充边界类型
"""
if __name__ == '__main__':
    filename = r'E:\testimage\lena.jpg'
    img = cv.imread(filename)
    img_gauss = cv.GaussianBlur(img, (3, 3), 1)
    cv.imshow("img", img)
    cv.imshow("imgGuass", img_gauss)
    cv.waitKey()
    cv.destroyAllWindows()

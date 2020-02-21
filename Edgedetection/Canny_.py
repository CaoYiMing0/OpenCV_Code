'''
①高斯模糊 - GaussianBlur
②灰度转换 - cvtColor
③计算梯度 – Sobel/Scharr
④非最大信号抑制
⑤高低阈值输出二值图像——高低阈值比值为2:1或3:1最佳
'''
# Canny算子
import cv2 as cv
def Canny_demo(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    #cv.imshow("dwad",gray)
    gradx = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    grady = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge_output = cv.Canny(gradx, grady, 50, 150)
    # edge_output = cv.Canny(gray, 50, 150) 可以替代前三行
    cv.imshow("Canny Edge", edge_output)

    dst = cv.bitwise_and(image, image, edge_output)
    cv.imshow("Color Edge", dst)
filename = r'E:\testimage\lena.jpg'
src = cv.imread(filename)
Canny_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()

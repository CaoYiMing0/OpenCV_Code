"""
局部阈值分割(自适应分割方法)
"""
import cv2 as cv

if __name__ == '__main__':
    filename = r'E:\testimage\lena.jpg'
    img = cv.imread(filename, 0)
    img_threshold = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 10)
    cv.imshow("img", img)
    cv.imshow("imgThreshold", img_threshold)
    cv.waitKey()
    cv.destroyAllWindows()

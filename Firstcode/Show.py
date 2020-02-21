import cv2 as cv

filename = r'E:\testimage\lona.jpg'
img = cv.imread(filename)

imgGauss = cv.GaussianBlur(img,(5,5),0);

cv.imshow("img",img)
cv.imshow("imgGuass",imgGauss)
cv.waitKey()
cv.destroyAllWindows()
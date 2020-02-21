"""
分水岭方法分割
原理与代码可见官方文档
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 我们从找到硬币的近似估计开始。为此，我们可以使用Otsu的二值化。
    img = cv2.imread('E:\\testimage\\coins.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("1.thresh", thresh)

    # 噪声消除
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow("2.morphologyEx", opening)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imshow("3.sure_background", sure_bg)

    # 寻找目标区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    cv2.imshow("4.sure_foreground", sure_fg)

    # 寻找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imshow("5.unknown", unknown)

    """
    cv2.connectedComponents（）它用0标记图像的背景，然后其他对象用从1开始的整数标记。
    但是我们知道，如果背景标记为0，则分水岭会将其视为未知区域。所以我们想用不同的整数来标记它。
    相反，我们将用定义的未知区域标记unknown为0。
    其实只有这样才能确定"分水岭"
    """
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    cv2.imshow("6", markers)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    cv2.imshow("7", markers)

    # 现在进行最后一步，应用分水岭。被标记的图像会被修改。边界标记为-1.
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv2.imshow("8.watershed", markers)
    cv2.imshow("9.final", img)

    cv2.waitKey()
    cv2.destroyAllWindows()
"""
Harris Corner Detection(哈里斯角点检测)
Harris算子原理:
    · 在灰度变化平缓区域，窗口内像素灰度积分近似保持不变
    · 在边缘区域，边缘方向：灰度积分近似不变，其余任意方向：剧烈变化；
    · 在角点处，任意方向均剧烈变化
    · λ₁和λ₂是椭圆的长短轴
        当λ₁和λ₂都比较小时，点（x,y）处于灰度变化平缓区域；
        当λ₁>>λ₂或λ₂>>λ₁时，点（x,y）为边界像素
        当λ₁和λ₂都比较大时，且近似相等时，点（x,y）为角点
    · 使用角点响应函数：
        R = detM - k(traceM)²
        traceM=λ₁+λ₂   detM = λ₁λ₂
        当R接近于零时，处于灰度变化平缓区域
        当R<0时，点为边界像素
        当R>0时，点为角点
dst = cv.cornerHarris( src, blockSize, ksize, k[, dst[, borderType]] )
    src，输入图像，即源图像，填Mat类的对象即可，且需为单通道8位或者浮点型图像
    dst，函数调用后的运算结果存在这里，即这个参数用于存放Harris角点检测的输出结果，和源图片有一样的尺寸和类型。
    blockSize，角点检测的邻域的大小。
    ksize，表示Sobel()算子的孔径大小。
    k，Harris参数。
    borderType，图像像素的边界模式，注意它有默认值BORDER_DEFAULT。更详细的解释，参考borderInterpolate函数。

Corner with SubPixel Accuracy详见官方文档（不知具体用处在哪）
    有时，您可能需要以最大的精度找到拐角。OpenCV带有一个函数cv2.cornerSubPix（），该函数可以进一步细化以亚像素精度检测到的角。
    参考代码：
    import cv2
    import numpy as np

    filename = 'chessboard2.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]

    cv2.imwrite('subpixel5.png',img)
"""

import cv2
import numpy as np

if __name__ == '__main__':
    filename = r'E:\testimage\black_white_bg.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # result用于标记角点，但不重要
    dst = cv2.dilate(dst, None)
    # 最佳阈值，它可能因图像而异。
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('dst', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
# Corner with SubPixel Accuracy
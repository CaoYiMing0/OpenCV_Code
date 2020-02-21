"""
Hough变换:
    · 采用参数空间变换的方法，对噪声和不间断直线的检测具有鲁棒性
    · 可用于检测圆和其他参数形状
    · 核心思想：直线y=kx+b每一条直线对应一个k, b,极坐标下对应一个点(ρ,θ)
步骤：
    · 将 空间量化成许多小格
    · 根据x-y平面每一个直线点代入θ的量化值，算出各个ρ，将对应格计数累加
    · 当全部点变换后，对小格进行检验。设置累计阈值T，计数器大于T的小格对应于共
      线点，其可以用作直线拟合参数。小于T的反映非共线点，丢弃不用
"""

import cv2
import numpy as np
if __name__ == '__main__':

    img = cv2.imread('E:\\testimage\\lena.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)

    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength, maxLineGap)
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('hough', img)
    cv2.waitKey(0)
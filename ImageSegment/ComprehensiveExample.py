"""
米粒的标记
"""
import cv2
import copy

if __name__ == '__main__':
    # 原始文件读入
    filename = r'E:\testimage\rice.jpg'
    image = cv2.imread(filename)

    # 三通道彩色图像转化为灰度的
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 大津算法阈值化
    _, bw = cv2.threshold(gray, 0, 0xff, cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # 形态学运算，去噪，减少米粒粘连
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, element)

    seg = copy.deepcopy(bw)
    # 各个区域对应的轮廓返回到cnts中
    bin, cnts, hier = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    # 对所有轮廓进行一个循环
    for i in range(len(cnts), 0, -1):
        c = cnts[i - 1]
        area = cv2.contourArea(c)
        # 如果面积小于10，不计入，可能是噪声
        if area < 10:
            continue
        count = count + 1
        print("blob", i, ":", area)

        x, y, w, h = cv2.boundingRect(c)
        # 原始图像上画出矩形和文本信息
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0xff), 1)
        cv2.putText(image, str(count), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0xff, 0))
    print("米粒数量:", count)
    cv2.imshow("源图", image)
    cv2.imshow("阈值化图", bw)
    cv2.waitKey()
    cv2.destroyAllWindows()

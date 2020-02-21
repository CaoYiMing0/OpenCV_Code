"""
使用光流法跟踪给定视频或摄像头中的运动特征点
步骤：
    1.视频采集
    2.图像预处理
    3.提取特征点
    4.使用光流法估计特征点运动
    5.相邻帧及特征点交换
提取特征点可以使用harris角点检测算子，检测是不是好的特征点可以使用
cv2.calcOpticalFlowPyrLK(image,corners,maxCorners,qualityLevel,minDistance)
    image:8位或32位浮点型输入图像，单通道。
    corners：保存检测出的角点。
    maxCorners：角点数目最大值，如果实际检测的角点超过此值，则只返回前maxcorners个强角点
    qualityLevel：角点的品质因子，决定角点可信度。
    minDistance:此邻域范围内如果存在更强角点，则删除此角点（一般一个角点附近不会存在其他角点）


效果较好
"""
import cv2

videoFileName = r'E:\\testimage\\videoplayback.mp4'
# 角点检测参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
#lucas kanade光流法参数
lk_parms = dict( winSize = (15,15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))
cap = cv2.VideoCapture(videoFileName)

#计算第一帧特征点
#光流以相邻帧来检测的，必须先拿到一帧
ret,prev = cap.read()
prevGray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY) #opencv目前只支持灰度图像的光流检测
p0 = cv2.goodFeaturesToTrack(prevGray,mask=None,**feature_params) #拿到比较好的特征点


while True:
    ret, frame = cap.read()
    if not ret:  # 没读到当前帧，结束
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #计算光流
    #p1为检测到的光流点的列表
    #st描述了每一个特征点计算光流时候的置信度
    p1,st,err = cv2.calcOpticalFlowPyrLK(prevGray,gray,p0,None,**lk_parms)

    #选取好的跟踪点
    goodPoints = p1[st==1]
    goodPrevPoints = p0[st==1]

    #在结果图像迭加画出特征点和计算出来的光流向量
    res = frame.copy()
    drawColor = (0,0,255)
    for i,(cur,prev) in enumerate(zip(goodPoints,goodPrevPoints)):
        x0,y0 = cur.ravel()
        x1,y1 = prev.ravel()
        cv2.line(res,(x0,y0),(x1,y1),drawColor)
        cv2.circle(res,(x0,y0),3,drawColor)
    # 更新上一帧
    prevGray = gray.copy()
    p0 = goodPoints.reshape(-1,1,2)

    #显示计算结果图像
    cv2.imshow("detect result",res)

    key = cv2.waitKey(30)  # 每一帧间隔30ms
    if key == 27:  # 按下ESC键，退出
        break

cap.release()
cv2.destroyAllWindows()

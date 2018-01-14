
# -*- coding:utf-8 -*-

import logging
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import unittest

class OPTChapter(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

    def showImageAndWaitClose(self, imgName, img):
        usematplotlib = False
        if usematplotlib:
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img2)
            plt.xticks([]), plt.yticks([])  # 不显示x、y轴坐标
            plt.show()
        else:
            while True:
              cv2.imshow(imgName, img)
              if cv2.waitKey(20) & 0xFF == 27: # 等待按键，如果为ESC则跳出循环
                break
            cv2.destroyAllWindows() # 销毁窗体

class Chapter4(OPTChapter):
    ''' 显示图片 '''
    def case4(self):
        img = cv2.imread('images/SGDevX.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])  # 不显示x、y轴坐标
        plt.show()

class Chapter5(OPTChapter):
    ''' 控制摄像头 '''
    def setUp(self):
        OPTChapter.setUp(self)
        self.cap = cv2.VideoCapture(0)

    def tearDown(self):
        # 注意：要先销毁窗体再释放摄像头，否则会crash
        cv2.destroyAllWindows()
        self.cap.release()

    def case1(self):
        while(True):
            ret, frame = self.cap.read()

            cv2.imshow('frame', frame)
            # 等待25ms，如果没有q键被按下，则继续循环，通常<=25ms的延迟可以确保视频流畅
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def case3(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                # flipCode参数：1水平翻转，0垂直翻转，-1竖屏垂直翻转
                frame = cv2.flip(frame, 1)

                cv2.imshow('frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

class Chapter6(OPTChapter):
    ''' 基本绘图 '''
    def setUp(self):
        OPTChapter.setUp(self)
        self.img = np.zeros((512, 512, 3), np.uint8)

    def case1(self):
        # 画线，参数为：起点、终点坐标、颜色、线宽
        cv2.line(self.img, (0, 0), (100, 100), (255, 0, 0), 1)

        # 矩形，参数为：左上角、右下角坐标、颜色、线宽
        cv2.rectangle(self.img, (50, 100), (100, 150), (0, 255, 0), 1)
        cv2.rectangle(self.img, (150, 50), (200, 150), (0, 255, 0), -1)

        # 圆，参数为：圆心坐标、半径、颜色、线宽
        cv2.circle(self.img, (250, 50), 25, (0, 0, 255), 1)
        cv2.circle(self.img, (250, 100), 25, (0, 0, 255), -1)

        # 椭圆，参数为：中心坐标、长短轴长度、起始旋转角度、起始绘制角度、中止绘制角度
        cv2.ellipse(self.img, (400, 10), (50, 25), 0, 0, 180, (255, 0, 0), 1)
        cv2.ellipse(self.img, (400, 50), (50, 25), 0, 60, 180, (255, 0, 0), -1)
        cv2.ellipse(self.img, (400, 100), (50, 25), 30, 60, 180, (255, 0, 0), -1)

        # 多边形，参数为：顶点坐标、是否封闭、颜色、线宽
        pts = np.array([[10, 200], [20, 230], [70, 220], [50, 210]], np.int32)
        cv2.polylines(self.img, [pts.reshape(-1, 1, 2)], False, (255, 255, 255), 1)
        pts = np.array([[70, 180], [80, 230], [140, 220], [100, 210]], np.int32)
        cv2.polylines(self.img, [pts.reshape(-1, 1, 2)], True, (255, 255, 255), 1)
        # 不加中括号，则只绘制顶点
        pts = np.array([[120, 180], [130, 230], [190, 220], [150, 210]], np.int32)
        cv2.polylines(self.img, pts.reshape(-1, 1, 2), True, (255, 255, 255), 2)

        # 文字
        font = cv2.FONT_HERSHEY_SIMPLEX                             # 文字
        cv2.putText(self.img, 'OpenCV', (200, 180), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(self.img, 'OpenCV', (250, 220), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        self.showImageAndWaitClose('image', self.img)
       
class Chapter7(OPTChapter):
    ''' 处理鼠标事件 '''
    def setUp(self):
        OPTChapter.setUp(self)
        self.img = np.zeros((512, 512, 3), np.uint8)
        self.imgName = 'image'
        cv2.namedWindow(self.imgName)

    def getEventFlagsName(self, value):
        valueNameMap = {1:'EVENT_FLAG_LBUTTON', 2:'EVENT_FLAG_RBUTTON', 
        4:'EVENT_FLAG_MBUTTON', 8:'EVENT_FLAG_CTRLKEY', 
        16:'EVENT_FLAG_SHIFTKEY', 32:'EVENT_FLAG_ALTKEY'}
        name = ''
        for k, v in valueNameMap.items():
            if value & k != 0:
                if len(name) == 0:
                    name = v
                else:
                    name += '|%s' % v

        return name

    def getEventActionName(self, action):
        actionNameMap = ['EVENT_MOUSEMOVE', 'EVENT_LBUTTONDOWN', 
        'EVENT_RBUTTONDOWN', 'EVENT_MBUTTONDOWN', 'EVENT_LBUTTONUP', 
        'EVENT_RBUTTONUP', 'EVENT_MBUTTONUP', 'EVENT_LBUTTONDBLCLK', 
        'EVENT_RBUTTONDBLCLK', 'EVENT_MBUTTONDBLCLK', 'EVENT_MOUSEWHEEL', 
        'EVENT_MOUSEHWHEEL', ]
        return actionNameMap[action]

    def case1(self):
        ''' 打印和鼠标相关的所有事件 '''
        events = [i for i in dir(cv2) if 'EVENT' in i]
        for event in events:
            logging.info('%2d %s' % (getattr(cv2, event), event))

    def case2(self):
        ''' 演示鼠标回调各参数的含义 '''
        def mouseCallback(event, x, y, flags, param):
            ''' param参数个数可以有多个 '''
            logging.info('(%3d, %3d), event:%-24s, flags:%-24s' % (x, y, 
                self.getEventActionName(event), 
                self.getEventFlagsName(flags)))
            img = param
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

        cv2.setMouseCallback(self.imgName, mouseCallback, self.img)
        self.showImageAndWaitClose(self.imgName, self.img)

class Chapter8(OPTChapter):
    def case1(self):
        def trackbarCallback(trackbarValue):
            logging.debug(trackbarValue)

        img = np.zeros((300, 512, 3), np.uint8)
        cv2.namedWindow('image')

        cv2.createTrackbar('R', 'image', 0, 255, trackbarCallback)
        cv2.createTrackbar('G', 'image', 0, 255, trackbarCallback)
        cv2.createTrackbar('B', 'image', 0, 255, trackbarCallback)

        switch = '0:OFF\n1:ON'
        cv2.createTrackbar(switch, 'image', 0, 1, trackbarCallback)
        while True:
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            r = cv2.getTrackbarPos('R', 'image')
            g = cv2.getTrackbarPos('G', 'image')
            b = cv2.getTrackbarPos('B', 'image')
            s = cv2.getTrackbarPos(switch, 'image')

            if s == 0:
                img[:] = 0
            else:
                img[:] = [b, g, r]

        cv2.destroyAllWindows()

class Chapter9(OPTChapter):
    def case2(self):
        img = cv2.imread('images/SGDevX.jpg')
        if len(img.shape) == 3:
            logging.info('row:%d, col:%d, channels:%d' % img.shape)
        elif len(img.shape) == 2:
            logging.info('row:%d, col:%d' % img.shape)

        self.showImageAndWaitClose('image', img)

    def case3(self):
        ''' 图像ROI（Region of Interest） '''
        img = cv2.imread('images/messi5.jpg')
        ball = img[280:340, 330:390]
        img[273:333, 100:160] = ball
        self.showImageAndWaitClose('image', img)

    def case4(self):
        ''' 图像通道拆分、合并 '''
        img = cv2.imread('images/messi5.jpg')
        b = img[:, :, 0]    # 读出所有b通道颜色
        logging.info(b)
        img[:, :, 2] = 0    # 将所有r通道颜色置0
        self.showImageAndWaitClose('image', img)

class Chapter13(OPTChapter):
    def case1(self):
        flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
        outputString = ''
        idx = 0
        for flag in flags:
            outputString += '%3d %-24s ' % (getattr(cv2, flag), flag)
            idx += 1
            if idx % 4 == 0:
                outputString += '\n'
        logging.info(outputString)

    def case2(self):
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # 转换到HSV

            lower_blue = np.array([110, 50, 50])            # 设定蓝色阈值
            upper_blue = np.array([130, 255, 255])

            mask = cv2.inRange(hsv, lower_blue, upper_blue) # 根据阈值构建掩模

            res = cv2.bitwise_and(frame, frame, mask=mask)  # 对原图和掩模图做位运算

            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            cv2.imshow('res', res)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

class Chapter14(OPTChapter):
    def case1(self):
        img = cv2.imread('images/SGDevX.jpg')
        # 这两种缩放图片的方式效果是一样的：
        # ①
        res1 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        self.showImageAndWaitClose('res1', res1)
        # ②
        height, width = int(height * 0.5), int(width * 0.5)
        res2 = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        self.showImageAndWaitClose('res2', res2)

    def case2(self):
        ''' 平移 '''
        img = cv2.imread('images/SGDevX.jpg')
        rows, cols = img.shape[:2]
        M = np.float32([[1, 0, 100], [0, 1, 50]]) # 变换矩阵tx, ty
        dst = cv2.warpAffine(img, M, (cols, rows))# 参数3 输出图像的尺寸
        self.showImageAndWaitClose('image', dst)

    def case3(self):
        ''' 旋转 '''
        img = cv2.imread('images/SGDevX.jpg')
        rows, cols = img.shape[:2]
        # 参数1 旋转中心，参数2 旋转角度，参数3 旋转后的缩放因子
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.6)

        dst = cv2.warpAffine(img, M, (cols, rows))
        self.showImageAndWaitClose('image', dst)

    def case4_5(self):
        ''' 仿射变换是将一张直视的图片沿x（如放倒）或y（如开关门）轴旋转。透视变换是仿射变换的逆过程，把放倒的照片立起来 '''
        img = cv2.imread('images/SGDevX.jpg')
        rows, cols = img.shape[:2]
        # ①仿射变换
        # 指定3个点，变换前后的坐标
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M1 = cv2.getAffineTransform(pts1, pts2)  # 组织变换矩阵
        dst1 = cv2.warpAffine(img, M1, (cols, rows))

        self.showImageAndWaitClose('image', img)
        self.showImageAndWaitClose('dst1', dst1)
        # ②透视变换
        pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250], [290, 200]])
        M2 = cv2.getPerspectiveTransform(pts2, pts1)
        dst2 = cv2.warpPerspective(dst1, M2, (cols, rows))
        self.showImageAndWaitClose('dst2', dst2)

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest samples.Chapter4.case4

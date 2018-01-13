
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

class Chapter4(OPTChapter):
    def case4(self):
        img = cv2.imread('images/SGDevX.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])  # 不显示x、y轴坐标
        plt.show()

class Chapter5(OPTChapter):
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
    def setUp(self):
        OPTChapter.setUp(self)
        self.img = np.zeros((512, 512, 3), np.uint8)

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
        
if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest samples.Chapter4.case4

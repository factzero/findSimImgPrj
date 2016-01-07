# -*- coding:UTF8 -*-
import os
import sys
import cv2
import numpy as np


def calcHistAndNorm(cvImg):
    bins = np.arange(256).reshape(256, 1)
    Hist = np.zeros((256, 256, 3))
    if 2 == len(cvImg.shape):
        Color = [(255, 255, 255)]
    elif 3 == cvImg.shape[2]:
        Color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for ch, col in enumerate(Color):
        HistItem = cv2.calcHist([cvImg], [ch], None, [256], [0, 256])
        cv2.normalize(HistItem, HistItem, 0, 255, cv2.NORM_MINMAX)
        HistInt = np.int32(np.around(HistItem))
        pts = np.int32(np.column_stack((bins, HistInt)))
        cv2.polylines(Hist, [pts], False, col)

    HistRes = np.flipud(Hist)
    return HistRes


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    InPath = r'/home/factzero/Pictures/girl'
    uInPath = unicode(InPath, 'utf8')
    ShowW = 256
    ShowH = 256
    cv2.namedWindow('Img', cv2.WINDOW_AUTOSIZE)
    for filename in os.listdir(uInPath):
        Fullfile = os.path.join(uInPath, filename)
        print Fullfile
        Fullfile = Fullfile.decode('utf8').encode('gbk')
        cvImgOri = cv2.imread(Fullfile)
        cvImgShow = cv2.resize(cvImgOri, (ShowW, ShowH))
        HistRGB = calcHistAndNorm(cvImgShow)

        cv2.imshow('Img', cvImgShow)
        cv2.imshow('ImgHist', HistRGB)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

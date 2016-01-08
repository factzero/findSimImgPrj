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
    cv2.namedWindow('ImgAll', cv2.WINDOW_AUTOSIZE)
    ImgAll = np.zeros((ShowH*2, ShowW*2, 3), np.uint8)
    for filename in os.listdir(uInPath):
        Fullfile = os.path.join(uInPath, filename)
        print Fullfile
        Fullfile = Fullfile.decode('utf8').encode('gbk')
        cvImgOri = cv2.imread(Fullfile, cv2.IMREAD_COLOR)
        cvImgShow = cv2.resize(cvImgOri, (ShowW, ShowH))
        HistRGB = calcHistAndNorm(cvImgShow)
        ImgAll[0:ShowH, 0:ShowW, 0:3] = cvImgShow.copy()
        ImgAll[0:ShowH, ShowW:ShowW*2, 0:3] = HistRGB.copy()
        cv2.putText(ImgAll, 'SRC', (0, 16), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), thickness = 1)
        cv2.putText(ImgAll, 'RGB Hist', (ShowW, 16), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), thickness = 1)

        cv2.imshow('ImgAll', ImgAll)
        cv2.waitKey(300)

    cv2.destroyAllWindows()

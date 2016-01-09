# -*- coding:UTF8 -*-
import os
import sys
import cv2
import numpy as np


def calcHistAndNorm(cvImg):
    bins = np.arange(256).reshape(256, 1)
    HistImg = np.zeros((256, 256, 3))
    Hist = np.zeros((256*3))
    if 2 == len(cvImg.shape):
        Color = [(255, 255, 255)]
    elif 3 == cvImg.shape[2]:
        Color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for ch, col in enumerate(Color):
        HistItem = cv2.calcHist([cvImg], [ch], None, [256], [0, 256])
        cv2.normalize(HistItem, HistItem, 0, 255, cv2.NORM_MINMAX)
        HistInt = np.int32(np.around(HistItem))
        Hist[256*ch:256*(ch+1)] = HistInt[0:256, 0]
        pts = np.int32(np.column_stack((bins, HistInt)))
        cv2.polylines(HistImg, [pts], False, col)

    HistRes = np.flipud(HistImg)
    return HistRes, Hist


def calcHistSimilar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - float(abs(l - r))/(max(l, r) + 1) for l, r in zip(lh, rh))/len(lh)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    InPath = r'/home/factzero/Pictures/girl'
    TestImg = r'/home/factzero/Pictures/2.jpg'
    uInPath = unicode(InPath, 'utf8')
    ShowW = 256
    ShowH = 256
    cv2.namedWindow('ImgAll', cv2.WINDOW_AUTOSIZE)
    ImgAll = np.zeros((ShowH*2, ShowW*2, 3), np.uint8)
    cvTestImg = cv2.imread(TestImg, cv2.IMREAD_COLOR)
    cvTestImg = cv2.resize(cvTestImg, (ShowW, ShowH))
    HistTstImg, HistTstRGB = calcHistAndNorm(cvTestImg)
    ImgAll[0:ShowH, 0:ShowW, 0:3] = cvTestImg.copy()
    ImgAll[0:ShowH, ShowW:ShowW*2, 0:3] = HistTstImg.copy()
    cv2.putText(ImgAll, 'TEST', (0, 16), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), thickness=1)
    cv2.putText(ImgAll, 'RGB Hist', (ShowW, 16), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), thickness=1)
    for filename in os.listdir(uInPath):
        Fullfile = os.path.join(uInPath, filename)
        print Fullfile
        Fullfile = Fullfile.decode('utf8').encode('gbk')
        cvImgOri = cv2.imread(Fullfile, cv2.IMREAD_COLOR)
        cvImgShow = cv2.resize(cvImgOri, (ShowW, ShowH))
        HistImg, HistRGB = calcHistAndNorm(cvImgShow)
        ImgAll[ShowH:ShowH*2, 0:ShowW, 0:3] = cvImgShow.copy()
        ImgAll[ShowH:ShowH*2, ShowW:ShowW*2, 0:3] = HistImg.copy()
        HistSim = calcHistSimilar(HistTstRGB, HistRGB)
        print HistSim
        cv2.putText(ImgAll, 'SRC', (0, ShowH + 16), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), thickness=1)
        cv2.putText(ImgAll, 'RGB Hist', (ShowW, ShowH + 16), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), thickness=1)

        cv2.imshow('ImgAll', ImgAll)
        cv2.waitKey(100)

    cv2.destroyAllWindows()

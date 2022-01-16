from PyQt5.QtGui import QImage, QPixmap       
import cv2 as cv
from PyQt5 import QtCore

def openCVtoQImage(cvImg):
        h, w = cvImg.shape[:2]
        qImg = QImage(cvImg.data, w, h, 3 * w, QImage.Format_RGB888) 
        return qImg

def scaleImage(cvImg, scalePercent, h, w):
        img = cvImg.copy() 
        dim = (int(img.shape[1] * scalePercent / 100), int(img.shape[0] * scalePercent / 100))
        pixmap = QPixmap(openCVtoQImage(cv.resize(img, dim, interpolation=cv.INTER_LINEAR_EXACT)))
        pixmap = pixmap.scaled(int(h * (scalePercent / 100)), int(w * (scalePercent / 100)), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        return pixmap

def cropImage(cvImg, x1, x2, y1, y2):
        img = cvImg.copy()
        dim = (int(img.shape[1]), int(img.shape[0]))
        # img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR_EXACT)
        cropped = img[int(dim[1]*y1):int(dim[1]*y2), int(dim[0]*x1):int(dim[0]*x2)].copy()
        return cropped
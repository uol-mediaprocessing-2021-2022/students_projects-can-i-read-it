import sys
import cv2 as cv
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from skimage import transform
from imutils.object_detection import non_max_suppression
import time


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

def cropImage(cvImg, y1, x1, y2, x2, scalePercent):
        img = cvImg.copy()
        dim = (int(img.shape[1] * scalePercent / 100), int(img.shape[0] * scalePercent / 100))
        img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR_EXACT)
        cropped = img[y1 : y2, x1 : x2].copy()
        return cropped

def runAnalysis(img):
        preprocessed_list, preprocessed_names_list = preprocess_image(img)
        # Add the original image to the list

def preprocess_image(img):
        # Create a list to store all preprocessed versions
        preprocessed_list = {}
        # create a list that stores strings describing all preprocessed versions
        preprocessed_names_list = {}
        # Add the original image to the list
        preprocessed_list[0] = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        preprocessed_names_list[0] = "RGB"
        # Create a grayscale copy
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Add greyscale copy to preprocessed List
        img_gray = cv.merge((img_gray,img_gray,img_gray)) # Hack for 3 gray channels
        preprocessed_list[1] = img_gray
        preprocessed_names_list[1] = "Grayscale"
        # Create a binary copy using "smart" thresholding
        th, img_bin = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY),0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        img_bin = cv.merge((img_bin, img_bin, img_bin)) # Hack for 3 gray channels
        # Add binary copy to preprocessed List
        preprocessed_list[3] = img_bin
        preprocessed_names_list[3] = "Binary with OTSU"
        # Apply edge-detection using sobel filter on the grayscale picture
        scale = 1
        delta = 0
        ddepth = cv.CV_16S
        # Reduce noise using gaussian blur
        src = cv.GaussianBlur(img, (3, 3), 0)
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        # gradient in x direction
        grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        # gradient in direction 
        grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        grad = cv.merge((grad, grad, grad)) # Hack for 3 gray channels
        preprocessed_list[4] = grad
        preprocessed_names_list[4] = "Sobel + Gauss"
        # Apply edge-detection using laplacian filter on grayscale picture
        # source : https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
        ddepth = cv.CV_16S
        kernel_size = 3
        # the original bgr-picture was already blurred and converted to grayscale for edge-detection with sobel  
        # we can now apply the laplace function to this picture
        dst = cv.Laplacian(gray, ddepth, ksize = kernel_size)
        dst = cv.convertScaleAbs(dst)
        dst = cv.merge((dst, dst, dst)) # hack for 3 gray channels
        preprocessed_list[5] = dst
        preprocessed_names_list[5] = "Laplace + Gauss"
        return (preprocessed_list, preprocessed_names_list)






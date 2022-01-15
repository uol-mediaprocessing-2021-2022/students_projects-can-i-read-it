import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
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

def cropImage(cvImg, x, y, w, h, scalePercent):
        img = cvImg.copy()
        dim = (int(img.shape[1] * scalePercent / 100), int(img.shape[0] * scalePercent / 100))
        img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR_EXACT)
        cropped = img[y : y + h, x : x + w].copy()
        return cropped

def runAnalysis(img):
        h, w = img.shape[:2]
        preprocessed_list, preprocessed_names_list = preprocess_image(img.copy())
        # perform radon_transform to the pictures
        M, h, w = get_radon_matrix(img.copy())
        preprocessed_list = rotate(preprocessed_list, M, h, w)
        boxes_list, rW, rH = east_detect(preprocessed_list)
        img_index = 2 # Set index of image used for text position detection
        # sorted_boxes = sort_boxes(boxes_list[img_index].tolist())   ---> sort_boxes method is never used in the actual app (Only useful in Notebook)
        connected_boxes = connect_boxes(boxes_list, img_index)
        tes_preprocess = img.copy()
        # Parameters for tesseract preprocessing
        scale_percent = 100 # percent of original size
        blur_amount = 3 # how much blur to apply
        bordersize = 10 # size of border to add
        results = []
        found_text_psm7 = ""

        for index, elem in enumerate(connected_boxes):
                for index2, (startX, startY, endX, endY) in enumerate(elem):
    
                        # Calculate adjusted margins 
                        y_start = int(startY * rH)
                        y_end = int(endY * rH)
                        x_start = int(startX * rW)
                        x_end = int(endX * rW)

                        cropped_img = tes_preprocess[y_start:y_end, x_start:x_end]
                        cropped_img = preprocess_tsrct(cropped_img, scale_percent, blur_amount, bordersize)

                        # Configuration setting for converting image to string
                        configuration = ("-l deu+eng --oem 1 --psm 7")  

                        # This will recognize the text from the image of the bounding box
                        d = pytesseract.image_to_data(cropped_img, output_type=pytesseract.Output.DICT, config=configuration)

                        text = ""

                        for index3, word in enumerate(d['text']):
                                if int(float(d['conf'][index3])) >= 50:
                                        text = text + word + " "

                        # Append detected text to current line of text and clear unwanted symbols
                        text = text.replace("|","")
                        text = text.replace("ı","")

                # Append line to the list of detected lines
                found_text_psm7 += text + '\n'
                results.append(text.rstrip())
        for line in results:
                print(line + '\n')

def preprocess_image(img): # Create a list to store all preprocessed versions
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
        preprocessed_list[2] = img_bin
        preprocessed_names_list[2] = "Binary with OTSU"
        return (preprocessed_list, preprocessed_names_list)

# gets the rotation matrix from the image
def get_radon_matrix(img):
        radon_preprocess = img.copy()
        I = cv.cvtColor(radon_preprocess, cv.COLOR_BGR2GRAY)
        h, w = I.shape
        #  fig = plt.figure(figsize=(8,8))
        # plt.imshow(I)
        # If the resolution is high, resize the image to reduce processing time.
        if (w > 640):
                 I = cv.resize(I, (640, int((h / w) * 640)))
        I = I - np.mean(I)  # Demean; make the brightness extend above and below zero

        # Do the radon transform
        sinogram = transform.radon(I)
        # Find the RMS value of each row and find "busiest" rotation,
        # where the transform is lined up perfectly with the alternating dark
        # text and white lines
        r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
        rotation = np.argmax(r)
        print('Rotation: {:.2f} degrees'.format(90 - rotation))

        # Rotate and save with the original resolution
        M = cv.getRotationMatrix2D((w/2, h/2), 90 - rotation, 1)
        return (M, h, w)

#returns List containing preprocessed and rotated images
def rotate(preprocessed_list, M, h, w):
        for idx, _ in enumerate(preprocessed_list):
                preprocessed_list[idx] = cv.warpAffine(preprocessed_list[idx].copy(), M, (w, h))
        return preprocessed_list


def east_detect(preprocessed_list):
        # Create list that will contain cordinates of found text boxes for each image
        boxes_list = []
        # Set min confidence for text detection
        min_score = 0.99 # Adjustable
        # Set scaling resolution, must be multiple of 32 for EAST
        res = 1280 # 1280x1280 works good
        for index, img_idx in enumerate(preprocessed_list):
                # Create copy and grab image dimensions
                orig = preprocessed_list[img_idx].copy()
                orig_rectangles = preprocessed_list[img_idx].copy()
                (H, W) = orig.shape[:2]

                # Adjust image dimensions
                (newW, newH) = (res, res)
                rW = W / float(newW)
                rH = H / float(newH)

                preprocessed_img = cv.resize(preprocessed_list[img_idx], (newW, newH))
                (H, W) = preprocessed_img.shape[:2]

                # 2 layers are needed, the confiidences of the predictions and koordinates of found Text
                layerNames = [
                        "feature_fusion/Conv_7/Sigmoid",
                        "feature_fusion/concat_3"]

                # Load EAST
                print("[INFO] Initializing text detection with EAST [" + str(index + 1) + "/" + str(len(preprocessed_list)) + "]")
                net = cv.dnn.readNet(os.getcwd() + '/east/frozen_east_text_detection.pb')
                net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

                # Create a "blob" and generate output using the trained model
                blob = cv.dnn.blobFromImage(preprocessed_img, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
                start = time.time()
                net.setInput(blob)
                (scores, geometry) = net.forward(layerNames)
                end = time.time()

                print("[INFO] text detection took {:.6f} seconds".format(end - start))
 
                # Grab the number of rows and columns from the scores volume, then
                # initialize our set of bounding box rectangles and corresponding
                # confidence scores
                (numRows, numCols) = scores.shape[2:4]
                rects = []
                confidences = []

                # Iterate the rows
                for y in range(0, numRows):
                        # Extract the scores alongside the cordinates of found text
                        scoresData = scores[0, 0, y]
                        xData0 = geometry[0, 0, y]
                        xData1 = geometry[0, 1, y]
                        xData2 = geometry[0, 2, y]
                        xData3 = geometry[0, 3, y]
                        anglesData = geometry[0, 4, y]
                        # Iterate columns
                        for x in range(0, numCols):
                                # Ignore scores below the threshold
                                if scoresData[x] < min_score:
                                        continue
                                # Calculate offset factor as our resulting feature maps will
                                # be 4x smaller than the input image
                                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                                # Calculate rotation angle
                                angle = anglesData[x]
                                cos = np.cos(angle)
                                sin = np.sin(angle)
                                # Calculate width and height of the bounding box
                                h = xData0[x] + xData2[x]
                                w = xData1[x] + xData3[x]
                                # Caclulate cordinates of the bounding box
                                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                                startX = int(endX - w)
                                startY = int(endY - h)
                                # Add the bounding box coordinates and probability score to
                                # our respective lists
                                rects.append((startX, startY, endX, endY))
                                confidences.append(scoresData[x])
                
                # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
                boxes = non_max_suppression(np.array(rects), probs=confidences)

                # Iterate all boxes of an image
                for (startX, startY, endX, endY) in boxes:
                        # Rescale the boxes
                        startX = int(startX * rW)
                        startY = int(startY * rH)
                        endX = int(endX * rW)
                        endY = int(endY * rH)

                # Add boxes to list containing all boxes for an image
                boxes_list.append(boxes)
        return boxes_list, rW, rH

# Method for getting the center of a bounding box
def get_center(box):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return int(center_x), int(center_y)

# Sort each boxes array
def sort_boxes(to_sort):
        sorted_boxes_tmp = []

        to_sort = sorted(to_sort, key=lambda k: [k[1], k[0]])

        row = []

        # Find all words within the same line and append them into a list
        while(len(to_sort) > 0):
                for rect in to_sort:
                        if len(to_sort) > 0:
                                y_new = get_center(to_sort[0])[1]
                                y_range = range(rect[1], rect[3])

                                if y_new in y_range:
                                        row.append(rect)
    
                to_sort = [i for i in to_sort if i not in row] 

                to_sort = sorted(to_sort, key=lambda k: [k[1], k[0]])
                row = sorted(row, key=lambda k: [k[0], k[1]])
                sorted_boxes_tmp.append(row)
                row = []

        return sorted_boxes_tmp

# Method for checking whether a point is in a bounding box
def check_x_intersection(point, range):
        connect_range = 16
        i = 0
        while i <= connect_range:
                if point+i in range:
                        return True
                i = i + 2
        return False

# Method for calculating the coordinates of a point on the right side of a bounding box
def get_reach(box):
        return int(box[2]), int((box[1] + box[3]) / 2)

def connect_boxes(boxes_list, img_index):
        connect_range = 16 # Max distance between boxes that will be connected, must be multiple of 2
        height_correction = 20
        # List containing connected boxes
        connected_boxes = []
        rect_list = boxes_list[img_index].tolist()
        rect_list = sorted(rect_list, key=lambda k: [k[0], k[1]])


        # Check whether a bounding box has a close neighbor to it's right, in that case both are 'connected'
        while(len(rect_list) > 0):

                curr_rect = rect_list[0]
                rect_reach = get_reach(curr_rect)

                curr_x1 = curr_rect[0]
                curr_x2 = curr_rect[2]
                curr_y1 = curr_rect[1]
                curr_y2 = curr_rect[3]

                boxes_to_remove = [curr_rect]

                for rect in rect_list:
                        if check_x_intersection(rect_reach[0], range(rect[0], rect[2])) and rect_reach[1] in range(rect[1], rect[3]) and curr_y2-curr_y1-height_correction <= rect[3]-rect[1]:
                                curr_x2 = rect[2]
      
                                if rect[1] < curr_y1:
                                        curr_y1 = rect[1]
      
                                if rect[3] > curr_y2:
                                        curr_y2 = rect[3]
      
                                rect_reach = get_reach(rect)
                                boxes_to_remove.append(rect)

                        connected_boxes.append([curr_x1, curr_y1, curr_x2, curr_y2])
                        rect_list = [i for i in rect_list if i not in boxes_to_remove] 

        connected_boxes = sort_boxes(connected_boxes)
        return connected_boxes

# Method for preprocessing the image for Tesseract usage
def preprocess_tsrct(img, scale_percent, blur_amount, bordersize):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
    
        # Resize image
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        processed = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

        # Make sure that black text is on white background
        w_pix_num = np.sum(processed == 255)
        b_pix_num = np.sum(processed == 0)
        if b_pix_num > w_pix_num:
                processed = cv.bitwise_not(processed)
  
        # Smoothen the background to reduce noise, median blur seems to perform better
        # processed = cv.GaussianBlur(processed, (blur_amount,blur_amount), 0)
        processed = cv.medianBlur(processed,blur_amount)

        # Add white border to image
        processed = cv.copyMakeBorder(
        processed,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv.BORDER_CONSTANT,
        value=[255.0, 255.0, 255.0]) 

        return processed
from asyncio import create_subprocess_exec
from json import detect_encoding
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from skimage import transform
from imutils.object_detection import non_max_suppression
import time
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit

class textdetect:
        # The preprocessed image using a chosen method
        preprocessed_image = None

        # List that will contain cordinates of found text boxes for the image
        boxes_list = None

        # Scales for image resizing (for EAST)
        rW = None
        rH = None

        def __init__(self, img): 
                self.img = img   # TEST THE OUTPUT !!! IMSHOW? 

        def getPath(self):
                widget = QWidget()
                widget.__init__()

                widget.title = 'PyQt5 input dialogs - pythonspot.com'
                widget.left = 10
                widget.top = 10
                widget.width = 640
                widget.height = 480

                text, okPressed = QInputDialog.getText(widget, "Enter path to tesseract.exe","", QLineEdit.Normal, "C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
                if okPressed and text != '':
                        pytesseract.pytesseract.tesseract_cmd = text

        def set_image(self, img):
                self.img = img

        # 'Hack' method for 3 channels (required later)
        def merge_preprocessed(self, img):
                img_merged = cv.merge((img, img, img))
                return img_merged

        # Greyscale the image
        def greyscale(self):
                img_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
                self.preprocessed_image = self.merge_preprocessed(img_gray)

        # Change contrast                       
        def contrast(self):
                alpha = 2 # Contrast
                beta = 0 # Brightness

                self.greyscale()
                self.preprocessed_image = cv.convertScaleAbs(self.preprocessed_image, alpha=alpha, beta=beta)
        
        # Create a binary copy using "smart" thresholding, also apply Gaussian blur for noise reduction
        def threshold(self):
                th, img_bin = cv.threshold(cv.cvtColor(self.img, cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                self.preprocessed_image = self.merge_preprocessed(img_bin)
        
        # Apply edge-detection using sobel filter on the grayscale picture
        def sobel(self):
                scale = 1
                delta = 0
                ddepth = cv.CV_16S
                # Reduce noise using gaussian blur
                src = cv.GaussianBlur(self.img, (3, 3), 0)
                gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
                # gradient in x direction
                grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
                # gradient in direction 
                grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
                
                abs_grad_x = cv.convertScaleAbs(grad_x)
                abs_grad_y = cv.convertScaleAbs(grad_y)
                
                grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

                self.preprocessed_image = self.merge_preprocessed(grad)
        
        # Apply edge-detection using laplacian filter on grayscale picture
        # Source : https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
        def laplace(self):
                ddepth = cv.CV_16S
                kernel_size = 3
                src = cv.GaussianBlur(self.img, (3, 3), 0)
                gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
                # Apply the laplace function to this picture
                dst = cv.Laplacian(gray, ddepth, ksize = kernel_size)
                dst = cv.convertScaleAbs(dst)

                self.preprocessed_image = self.merge_preprocessed(dst)

        # Deskew the image using Radon Transformation
        # Source: https://stackoverflow.com/questions/46084476/radon-transformation-in-python/55454222
        def use_radon_rotation(self):
                # Here we REQUIRE the image to be cropped to in the best case ONLY contain the
                # text that needs to be rotated -> requires cropping in most cases
                radon_preprocess = self.img.copy()
                I = cv.cvtColor(radon_preprocess, cv.COLOR_BGR2GRAY)
                h, w = I.shape
                #I = I[0:3000, 400:2200] #<- Don't need this, we assume the image was already cropped by an user

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
                print('[INFO] ' + 'Rotation: {:.2f} degrees'.format(90 - rotation))

                # Rotate and save with the original resolution
                M = cv.getRotationMatrix2D((w/2, h/2), 90 - rotation, 1)
                self.preprocessed_image = cv.warpAffine(self.preprocessed_image, M, (w, h))
                self.img = cv.warpAffine(self.img, M, (w, h))
                

        # Use EAST to detect text in the image
        # Source: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
        def east_detect(self):
                # Set min confidence for text detection
                min_score = 0.99 # Adjustable

                # Set scaling resolution, must be multiple of 32 for EAST
                res = 1280 # 1280x1280 works good

                # Create copy and grab image dimensions
                orig = self.preprocessed_image.copy()
                (H, W) = orig.shape[:2]

                # Adjust image dimensions
                (newW, newH) = (res, res)
                self.rW = W / float(newW)
                self.rH = H / float(newH)

                resized_img = cv.resize(self.preprocessed_image.copy(), (newW, newH))
                (H, W) = resized_img.shape[:2]

                # 2 layers are needed, the confiidences of the predictions and koordinates of found Text
                layerNames = [
                        "feature_fusion/Conv_7/Sigmoid",
                        "feature_fusion/concat_3"]

                # Load EAST
                print("[INFO] Initializing text detection with EAST [" + str(1) + "/" + str(1) + "]")
                net = cv.dnn.readNet(os.getcwd() + '/east/frozen_east_text_detection.pb')
                net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

                # Create a "blob" and generate output using the trained model
                blob = cv.dnn.blobFromImage(resized_img, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
                start = time.time()
                net.setInput(blob)
                (scores, geometry) = net.forward(layerNames)
                end = time.time()

                print("[INFO] EAST text detection took {:.6f} seconds".format(end - start))
        
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
                
                self.boxes_list = boxes

        # Method for getting the center of a bounding box
        def get_center(self, box):
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                return int(center_x), int(center_y)
                
        # Method for calculating the coordinates of a point on the right side of a bounding box
        def get_reach(self, box):
                return int(box[2]), int((box[1] + box[3]) / 2)

        # Sort each boxes array
        def sort_boxes(self, to_sort):
                sorted_boxes_tmp = []

                to_sort = sorted(to_sort, key=lambda k: [k[1], k[0]])

                row = []

                # Find all words within the same line and append them into a list
                while(len(to_sort) > 0):

                        for rect in to_sort:
                                if len(to_sort) > 0:

                                        y_new = self.get_center(to_sort[0])[1]
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
        def check_x_intersection(self, point, range, connect_range):
                i = 0
                while i <= connect_range:
                        if point + i in range:
                                return True
                        i = i + 2
                return False

        # Connect and sort text boxes
        def sort_and_connect(self):
                # TODO Add theese changes to notebook = connect range scaling
                connect_range = 200 # Max distance between boxes that will be connected, must be multiple of 2
                height_correction = 0.85 # Variable that prevents vertically larger boxes from connecting with smaller boxes

                # List containing connected boxes
                connected_boxes = []

                # Check if any text was found
                if len(self.boxes_list) == 0:
                        return 0

                rect_list = self.boxes_list.tolist()
                rect_list = sorted(rect_list, key=lambda k: [k[0], k[1]])

                # Check whether a bounding box has a close neighbor to it's right, in that case both are 'connected'
                while(len(rect_list) > 0):
                        curr_rect = rect_list[0]
                        rect_reach = self.get_reach(curr_rect)

                        curr_x1 = curr_rect[0]
                        curr_x2 = curr_rect[2]
                        curr_y1 = curr_rect[1]
                        curr_y2 = curr_rect[3]

                        boxes_to_remove = [curr_rect]

                        for rect in rect_list:
                                if self.check_x_intersection(rect_reach[0], range(rect[0], rect[2]), connect_range) and rect_reach[1] in range(rect[1], rect[3]) and abs(1 - abs((curr_y2-curr_y1) / (rect[3]-rect[1]))) <= height_correction:
                                        curr_x2 = rect[2]
                
                                        if rect[1] < curr_y1:
                                                curr_y1 = rect[1]
                        
                                        if rect[3] > curr_y2:
                                                curr_y2 = rect[3]
                        
                                        rect_reach = self.get_reach(rect)
                                        boxes_to_remove.append(rect)

                        connected_boxes.append([curr_x1, curr_y1, curr_x2, curr_y2])
                        rect_list = [i for i in rect_list if i not in boxes_to_remove] 

                # Draw the expanded boxes on this image
                orig_connected_rectangles = self.img.copy()

                # Iterate all boxes of an image
                for (startX, startY, endX, endY) in connected_boxes:
                        # Rescale the boxes
                        startX = int(startX * self.rW)
                        startY = int(startY * self.rH)
                        endX = int(endX * self.rW)
                        endY = int(endY * self.rH)
                        # Draw boxes in output image
                        cv.rectangle(orig_connected_rectangles, (startX, startY), (endX, endY), (0, 255, 0), 3)

             
                # TODO Display the image with boxes for interactivity
                # imshow(...)
                plt.imshow(orig_connected_rectangles)
                plt.show()

                # Sort the boxes
                self.boxes_list = self.sort_boxes(connected_boxes)

        # Method for preprocessing the image for Tesseract usage
        def preprocess_tsrct(self, imge, scale_percent, bordersize, blur_amount):
                width = int(imge.shape[1] * scale_percent / 100)
                height = int(imge.shape[0] * scale_percent / 100)
                dim = (width, height) # TODO WHy can this be 0 ???
                
                # Resize image
                resized = cv.resize(imge, dim, interpolation = cv.INTER_AREA)
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

        def extract_text(self):
                tes_preprocess = self.img.copy()
                
                # DEBUG
                # plt.imshow(tes_preprocess)
                # plt.show()

                # Parameters
                scale_percent = 100 # percent of original size
                bordersize = 10 # size of border to add
                blur_amount = 3 # how much blur to apply

                results = []
                found_text_psm7 = ""

                for index, elem in enumerate(self.boxes_list):

                        for index2, (startX, startY, endX, endY) in enumerate(elem):
                
                                # Calculate adjusted margins 
                                y_start = int(startY * self.rH)
                                y_end = int(endY *  self.rH)
                                x_start = int(startX *  self.rW)
                                x_end = int(endX *  self.rW)

                                cropped_img = tes_preprocess[y_start:y_end, x_start:x_end]
                                cropped_img = self.preprocess_tsrct(cropped_img, scale_percent, bordersize, blur_amount)

                                # Configuration setting for converting image to string
                                configuration = ("-l deu+eng --oem 1 --psm 7")  

                                # This will recognize the text from the image of the bounding box
                                d = pytesseract.image_to_data(cropped_img, output_type=pytesseract.Output.DICT, config=configuration)

                                text = ""

                                for index3, word in enumerate(d['text']):
                                        if float(d['conf'][index3]) >= 50:
                                                text = text + word + " "

                                # Append detected text to current line of text and clear unwanted symbols
                                text = text.replace("|","")
                                text = text.replace("ı","")

                        # Append line to the list of detected lines
                        found_text_psm7 += text + '\n'
                        results.append(text.rstrip())

                detected_text = ""
                for line in results:
                        detected_text = detected_text + line + '\n'

                return detected_text
                

        def runAnalysis(self, parameters): # TODO Create parameters in window py, take other params into account
                match parameters["preprocess_method"]:
                        case 1:
                                self.greyscale()
                        case 2:
                                self.contrast()
                        case 3:
                                self.threshold()
                        case 4:
                                self.sobel()
                        case 5:
                                self.laplace()        

                if parameters["enable_radon"] == True:
                        self.use_radon_rotation()
                
                detected_text = ""
                self.east_detect()
                found_boxes = self.sort_and_connect()

                if found_boxes != 0:
                        detected_text = self.extract_text()
                else:
                        detected_text = "No text has been detected." 
                print(detected_text)
                return detected_text
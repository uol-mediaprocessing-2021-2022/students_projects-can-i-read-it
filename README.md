# Mediaprocessing 2021/2022 - Can I Read It? 
-----------------------------------------------------------------------------------------------------
# Installation instructions:
To execute the gui-version of our project (window.py) you have to install the following dependencies:

* Python3.8 or higher (https://www.python.org/downloads/)
* Tesseract-OCR (https://tesseract-ocr.github.io/tessdoc/Downloads.html)
* PyQt5 (pip install PyQt5)
* OpenCV (pip install opencv-python)
* Numpy (pip install numpy)
* Matplotlib (pip install matplotlib)
* Python Imaging Library (pip install PIL-tools)
* Scikit image (pip install skimage)
* Python imutils (pip install imutils)
* Pytesseract (pip install pytesseract)

You might also have to install the micross font (provided in our github repository).

Please note that cropping images is not fully supported on macOS because they have removed
the support for sizegrips. We therefore recommend running this application under Linux or Windows.
------------------------------------------------------------------------------------------------------

# User Manual
You can start the gui-version with the command-line instruction ``` python window.py```, if this does not
work try ```python3 window.py```.

Upon startup the application will launch a small window headlined "Enter path to tesseract.exe", if you
are using a windows system you will have to provide the path to your tesseract installation, if you are
on a unix-based system and tesseract is added to your path, you can just close this window by pressing
the cancel button. After the small window is closed, the main window becomes usable. 

Let us take a look at the main windows user interface:

<img src="screenshot1.png" width="550" height="700">


* The purpose of the area in box 1 is to open and crop the image and to run the textdetection.
* The purpose of the area in box 2 is to allow the user to manually change the EAST settings
* The purpose of the area in box 3 is to allow the user to manually alter the tesseract preprocessing
* The purpose of the area in box 4 is to allow the user to change the tesseract operating mode and language 
* The area in box 5 will contain the detected text

We will now walk you through an exemplary user session of this application:

## Opening an image
To load a picture, you have to press the button labeled ```open```, this will open a file-dialogue where you 
can choose the picture.

## Cropping an image
After you have loaded the picture you want to analyze you may want to crop the relevant section of the image.
To do so press the button labeled ```crop``` once and resize and move the blue box that appears on the picture
according to your needs. If you now press ```crop``` again this will crop the selected region of the picture.
If you are not satisfied with the cropped image you can always return to the original picture by pressing the
```reset``` button.

## Run the analysis
After loading the picture (and optionally cropping it) you can run the text-detection by pressing ```Detect Text```.

## Adjust EAST settings
By using the dropdown menu in area 2 you can choose the preprocessing method for the EAST textbox detection. 
The box labeled with "EAST resolution" allows you to change the resolution that the textbox detection
works with, a higher resolution can improve results but can also increase the runtime significantly. 
Please note that the values for resolution have to be a multiple of 32 for the program to work.



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





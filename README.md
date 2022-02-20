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

------------------------------------------------------------------------------------------------------

# User Manual
You can start the gui-version with the command-line instruction ``` python window.py```, if this does not
work try ```python3 window.py```.

Upon startup the application will launch a small window headlined "Enter path to tesseract.exe", if you
are using a windows system you will have to provide the path to your tesseract installation, if you are
on a unix-based system and tesseract is added to your path, you can just close this window by pressing
the cancel button. After the small window is closed, the main window becomes usable. 

Let us take a look at the main windows user interface:


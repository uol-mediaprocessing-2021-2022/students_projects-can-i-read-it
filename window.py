from sqlite3 import paramstyle
import sys
import cv2 as cv
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication,  QPushButton, QFileDialog, QWidget, QHBoxLayout,  QSizeGrip, QRubberBand, QTextBrowser, QComboBox, QCheckBox, QLineEdit
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5 import uic, QtCore
from textdetect_library import *
from Textdetect import textdetect     

class MainWindow(QMainWindow):

    cropped = None

    def __init__(self):
        super(MainWindow, self).__init__()
     
        self.initUI()
        self.statusBar().hide()
        self.setContentsMargins(0, 0, 0, 0)
        self.textdetection = textdetect(None)
        self.textdetection.getPath()

    def initUI(self):
        # Load ui file
        uic.loadUi("window.ui", self)

        # Define widgets
        self.openButton = self.findChild(QPushButton, "openButton")
        self.runButton = self.findChild(QPushButton, "runButton")
        self.cropButton = self.findChild(QPushButton, "cropButton")
        self.resetCropButton = self.findChild(QPushButton, "resetCropButton")
        self.textWindow = self.findChild(QTextBrowser, "textWindow")
        self.imageLabel = self.findChild(QLabel, "imageLabel")
        self.background1 = self.findChild(QLabel, "background1")
        self.imageX = self.imageLabel.geometry().getCoords()[2]
        self.imageY = self.imageLabel.geometry().getCoords()[3]
        self.preprocessingBox = self.findChild(QComboBox, "preprocessingBox")
        self.radonCheck = self.findChild(QCheckBox, "radonCheck")
        self.scalingEdit = self.findChild(QLineEdit, "scalingEdit")
        self.blurEdit = self.findChild(QLineEdit, "blurEdit")
        self.borderEdit = self.findChild(QLineEdit, "borderEdit")
        self.confEdit = self.findChild(QLineEdit, "confEdit")
        self.ocrBox = self.findChild(QComboBox, "ocrBox")
        self.languageBox = self.findChild(QComboBox, "languageBox")
        self.tesMethodBox = self.findChild(QComboBox, "tesMethodBox")
        self.strengthEdit = self.findChild(QLineEdit, "strengthEdit")
        self.resEdit = self.findChild(QLineEdit, "resEdit")

        # Define methods
        self.openButton.clicked.connect(self.handleOpen)
        self.cropButton.clicked.connect(self.handleCrop)
        self.cropButton.setCheckable(True)
        self.runButton.clicked.connect(self.handleRun)
        self.resetCropButton.clicked.connect(self.resetCrop)

        # Ininitialize widgets
        self.preprocessingBox.addItems(["none","greyscale", "contrast", "threshold", "sobel", "laplace"])
        self.preprocessingBox.setCurrentIndex(3)
        self.languageBox.addItems(["deu", "eng", "deu+eng", "fra"])
        self.languageBox.setCurrentIndex(2)
        self.tesMethodBox.addItems(["threshold", "contrast", "grayscale"])
        self.tesMethodBox.setCurrentIndex(0)
        self.ocrBox.addItems(["psm1", "psm7"])
        self.ocrBox.setCurrentIndex(1)

        # Styles
        self.cropButton.setStyleSheet("background-color : lightgrey")
        self.background1.setStyleSheet("background-color : #ffdf99")
        self.imageLabel.setStyleSheet("background-color : #fff7e6")

        # Display app
        self.show()

    def handleOpen(self):
        # Open file dialog and load path of image
        image_path = QFileDialog.getOpenFileName(self, "Open Picture", "", "PNG Files (*.png);;JPG Files (*.jpg)")

        # Load picture to graphicsView
        if len(image_path[0]) > 0:
            self.cvOrig = cv.cvtColor(cv.imread(image_path[0]), cv.COLOR_BGR2RGB)
            self.load_image(self.cvOrig)
            self.textdetection.set_image(self.cvOrig) 

    def load_image(self, img):
        self.h, self.w = img.shape[:2]

        self.pixmap = QPixmap(openCVtoQImage(img))
        self.pixmap = self.pixmap.scaled(self.imageX, self.imageY, QtCore.Qt.KeepAspectRatio)

        self.bR_x = self.pixmap.rect().bottomRight().x()
        self.bR_y = self.pixmap.rect().bottomRight().y()

        self.imageLabel.setPixmap(self.pixmap)
        self.imageLabel.adjustSize()
        self.textWindow.setText("")

    def handleCrop(self):
        if self.cropButton.isChecked():
            self.band = ResizableRubberBand(self.imageLabel)
            self.band.setGeometry(0, 0, int(self.bR_x / 2), int(self.bR_y / 2))
            # Set background color back to light-grey
            self.cropButton.setStyleSheet("background-color : lightgrey")
        else:
            x, y, w, h = self.band.geometry().getCoords()

            x1 = x / self.bR_x
            x2 = w / self.bR_x
            y1 = y / self.bR_y
            y2 = h / self.bR_y

            self.cropped = cropImage(self.cvOrig.copy(), x1, x2, y1, y2)
            self.textdetection.set_image(self.cropped)
            self.pixmap = QPixmap(openCVtoQImage(self.cropped))
            self.pixmap = self.pixmap.scaled(self.imageX, self.imageY, QtCore.Qt.KeepAspectRatio)

            # Close the RubberBand
            self.band.close()

            self.imageLabel.setPixmap(self.pixmap)
            self.imageLabel.adjustSize()
            # setting background color to light-blue
            self.cropButton.setStyleSheet("background-color : lightblue")

    # DEFAULT PARAM
    default_param = {
        "preprocess_method" : 3,
        "enable_radon" : True,
        "scaling" : 100,
        "blur" : 3,
        "border" : 10,
        "minConf" : 30,
        "ocrMode" : "psm7",
        "east_res" : 1280,
        "strength" : "def",
        "tes_method" : "threshold",
        "languageMode" : "deu+eng"
    }
    
    def handleRun(self):
        print("Running analysis")
        param = self.getParam()
        detected_text, drawn_text = self.textdetection.runAnalysis(param)
        if len(detected_text) > 1 and drawn_text is not None:
            self.load_image(drawn_text)
            
        self.textWindow.setText(detected_text)
    
    def getParam(self):
        param = self.default_param
        param["preprocess_method"] = self.preprocessingBox.currentIndex()
        param["enable_radon"] = self.radonCheck.isChecked()
        param["scaling"] = self.scalingEdit.text()
        param["blur"] = self.blurEdit.text()
        param["border"] = self.borderEdit.text()
        param["minConf"] = self.confEdit.text()
        param["ocrMode"] = self.ocrBox.currentText()
        param["languageMode"] = self.languageBox.currentText()
        param["east_res"] = int(self.resEdit.text())
        param["strength"] = self.strengthEdit.text()
        param["tes_method"] = self.tesMethodBox.currentText()
        return param
        
    def resetCrop(self):   
        self.pixmap = QPixmap(openCVtoQImage(self.cvOrig))
        self.pixmap = self.pixmap.scaled(self.imageX, self.imageY, QtCore.Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(self.pixmap)
        self.imageLabel.adjustSize()
        self.textWindow.setText("")
        self.load_image(self.cvOrig)
        self.textdetection.close_windows()
        self.textdetection.set_image(self.cvOrig)


        
class ResizableRubberBand(QWidget):
    def __init__(self, parent=None):
        super(ResizableRubberBand, self).__init__(parent)

        self.draggable = True
        self.dragging_threshold = 5
        self.mousePressPos = None
        self.mouseMovePos = None
        self.borderRadius = 5

        self.setWindowFlags(QtCore.Qt.SubWindow)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            QSizeGrip(self), 0,
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(
            QSizeGrip(self), 0,
            QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self._band = QRubberBand(
            QRubberBand.Rectangle, self)
        self._band.show()
        self.show()

    def resizeEvent(self, event):
        self._band.resize(self.size())

    def paintEvent(self, event):
        # Get current window size
        window_size = self.size()
        qp = QPainter()
        qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        qp.drawRoundedRect(0, 0, window_size.width(), window_size.height(),
                           self.borderRadius, self.borderRadius)
        qp.end()

    def mousePressEvent(self, event):
        if self.draggable and event.button() == QtCore.Qt.RightButton:
            self.mousePressPos = event.globalPos()                # global
            self.mouseMovePos = event.globalPos() - self.pos()    # local
        super(ResizableRubberBand, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draggable and event.buttons() & QtCore.Qt.RightButton:
            globalPos = event.globalPos()
            moved = globalPos - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # Move when user drag window more than dragging_threshold
                diff = globalPos - self.mouseMovePos
                self.move(diff)
                self.mouseMovePos = globalPos - self.pos()
        super(ResizableRubberBand, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mousePressPos is not None:
            if event.button() == QtCore.Qt.RightButton:
                moved = event.globalPos() - self.mousePressPos
                if moved.manhattanLength() > self.dragging_threshold:
                    # Do not call click event or so on
                    event.ignore()
                self.mousePressPos = None
        super(ResizableRubberBand, self).mouseReleaseEvent(event)

# Initialize app
app = QApplication(sys.argv)
UIWindow = MainWindow()
app.exec_()
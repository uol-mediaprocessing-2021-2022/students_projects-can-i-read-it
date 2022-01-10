import sys
import cv2 as cv
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication,  QPushButton, QFileDialog, QWidget, QHBoxLayout,  QSizeGrip, QRubberBand
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5 import uic, QtCore
from textdetect_libary import *

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
        

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI() 

    def initUI(self):
        # load ui file
        uic.loadUi("window.ui", self)

        # define widgets
        self.openButton = self.findChild(QPushButton, "openButton")
        self.runButton = self.findChild(QPushButton, "runButton")
        self.cropButton = self.findChild(QPushButton, "cropButton")
        self.zoomInButton = self.findChild(QPushButton, "zoomInButton")
        self.zoomOutButton = self.findChild(QPushButton, "zoomOutButton")
        self.imageLabel = self.findChild(QLabel, "imageLabel")

        self.openButton.clicked.connect(self.handleOpen)
        self.cropButton.clicked.connect(self.handleCrop)
        self.runButton.clicked.connect(self.handleRun)
        self.zoomInButton.clicked.connect(self.zoomIn)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        # display app
        self.show()

    


    def handleOpen(self):
        # open file dialog and load path of image
        image_path = QFileDialog.getOpenFileName(self, "Open Picture", "", "PNG Files (*.png);;JPG Files (*.jpg)")

        # load picture to graphicsView
        if image_path:
            self.cvOrig = cv.cvtColor(cv.imread(image_path[0]), cv.COLOR_BGR2RGB)
            self.h, self.w = self.cvOrig.shape[:2]
            self.scalePercent = 100
            self.pixmap = QPixmap(openCVtoQImage(self.cvOrig))
            self.imageLabel.setPixmap(self.pixmap)
            self.imageLabel.adjustSize()

    def handleCrop(self):
        self.band = ResizableRubberBand(self.imageLabel)
        self.band.setGeometry(0, 0, 150, 150)

    def handleRun(self):
       print("running analysis") 
       if self.band:
           x1, y1, x2, y2 = self.band.geometry().getCoords()
           cropped = cropImage(self.cvOrig.copy(), y1, x1, y2, x2, self.scalePercent)
           self.pixmap = QPixmap(openCVtoQImage(cropped)) 
           self.imageLabel.setPixmap(self.pixmap)
           self.imageLabel.adjustSize()

    def zoomIn(self):   
       if self.scalePercent < 200:
            self.scalePercent += 10 
            self.imageLabel.setPixmap(scaleImage(self.cvOrig.copy(), self.scalePercent, self.h, self.w))
            self.imageLabel.adjustSize()
        
        
    def zoomOut(self):
        if self.scalePercent > 10:
            self.scalePercent -= 10 
            self.imageLabel.setPixmap(scaleImage(self.cvOrig, self.scalePercent, self.h, self.w))
            self.imageLabel.adjustSize()
 


# initialize app

app = QApplication(sys.argv)
UIWindow = MainWindow()
app.exec_()

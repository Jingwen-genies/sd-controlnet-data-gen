from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QGraphicsView

from avatar_generation.UI.keypoint import Keypoint


class CustomGraphicsView(QGraphicsView):
    zoomChanged = pyqtSignal(float)
    bboxModeChanged = pyqtSignal(bool)  # Define a new signal

    def __init__(self, parent=None):
        super(CustomGraphicsView, self).__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.scaleFactor = 1.0
        self.shouldFitInView = True

        self.isBboxMode = False
        self.startPoint = None
        self.currentBbox = None

        # For panning
        self._panning = False
        self._panStartX = 0
        self._panStartY = 0
        self.shiftPressed = False

    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Shift:
            self.shiftPressed = True
            print(f"Shift pressed: {self.shiftPressed}")

        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self.shiftPressed = False
            print(f"Shift released: {self.shiftPressed}")
        else:
            super().keyReleaseEvent(event)


    def mousePressEvent(self, event):
        print("\nreceived mouse press event")
        if event.button() == Qt.MiddleButton:
            print("middle button clicked")
            self._panning = True
            self._panStartX, self._panStartY = event.x(), event.y()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            print("RightButten clicked")
            self.isBboxMode = not self.isBboxMode
            print(f"bbox mode: {self.isBboxMode}")
            self.bboxModeChanged.emit(self.isBboxMode)
        elif event.button() == Qt.LeftButton and self.isBboxMode:
            print("LeftButten clicked and in bbox mode")
            self.startPoint = self.mapToScene(event.pos())
            if self.currentBbox:
                self.currentBbox = None
            self.currentBbox = Bbox(self.scene())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - (event.x() - self._panStartX))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - (event.y() - self._panStartY))
            self._panStartX, self._panStartY = event.x(), event.y()
        elif self.isBboxMode and self.startPoint:
            endPoint = self.mapToScene(event.pos())
            self.currentBbox.createOrUpdate(self.startPoint.x(), self.startPoint.y(), endPoint.x(), endPoint.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        elif self.isBboxMode:
            self.startPoint = None
        else:
            super().mouseReleaseEvent(event)

    def start_group_drag(self, index):
        self.groupToMove = []
        group = self.parent().landmarks.get_group(index)  # Make sure to replace with your actual landmarks object
        for idx in group:
            for item in self.scene().items():
                if isinstance(item, Keypoint) and item.index == idx:
                    self.groupToMove.append(item)

    def resizeEvent(self, event):
        if self.shouldFitInView and self.scene():
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def wheelEvent(self, event):
        self.shouldFitInView = False
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        minFactor = 0.2
        maxFactor = 10.0

        # get the wheel data, positive means zoom in, negative means zoom out
        if event.angleDelta().y() > 0:
            print("zoom in")
            zoomFactor = zoomInFactor
        else:
            print("zoom out")
            zoomFactor = zoomOutFactor

        newScaleFactor = self.scaleFactor * zoomFactor
        print(
            f"newScaleFactor: {newScaleFactor}, self.scaleFactor:{self.scaleFactor}, minFactor:{minFactor}, maxFactor: {maxFactor}")
        if newScaleFactor < minFactor or newScaleFactor > maxFactor:
            print(f"newScaleFactor: {newScaleFactor},minFactor:{minFactor}, maxFactor: {maxFactor}")
            print("Zoom factor out of range")
            return

        self.scale(zoomFactor, zoomFactor)
        self.scaleFactor = newScaleFactor

    def getViewWidth(self):
        return self.viewport().width()

    def getViewHeight(self):
        return self.viewport().height()

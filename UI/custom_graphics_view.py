from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QGraphicsView

from avatar_generation.UI.bbox import Bbox


class CustomGraphicsView(QGraphicsView):
    zoomChanged = pyqtSignal(float)
    bboxModeChanged = pyqtSignal(bool)  # Define a new signal

    def __init__(self, parent=None):
        super(CustomGraphicsView, self).__init__(parent)
        self.parent = parent
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.scaleFactor = 1.0
        self.shouldFitInView = True

        self.isBboxMode = False
        self.startPoint = None
        self.currentBbox = None

        self.isSelectionMode = False
        self.selectedRect = None
        self.isLeftMousePressed = False


        # For panning
        self._panning = False
        self._panStartX = 0
        self._panStartY = 0
        self.shiftPressed = False

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key_Shift:
            self.shiftPressed = True
            print(f"Shift pressed: {self.shiftPressed}")
        elif event.key() == Qt.Key_Z:
            self.isSelectionMode = True
            print(f"z pressed: {self.isSelectionMode}")

        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key_Shift:
            self.shiftPressed = False
            print(f"Shift released: {self.shiftPressed}")
        elif event.key() == Qt.Key_Z:
            self.isSelectionMode = False
            print(f"z released: {self.isSelectionMode}")
            self.selectedRect = None
            self.update()
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
        elif event.button() == Qt.LeftButton and self.isSelectionMode:
            self.isLeftMousePressed = True
            self.startPoint = self.mapToScene(event.pos())
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
        elif self.isSelectionMode and self.startPoint and self.isLeftMousePressed:
            endPoint = self.mapToScene(event.pos())
            self.selectedRect = QRectF(self.startPoint, endPoint)
            self.selectPointsInRectangle()  # Select points while dragging
            print(f"selected rect is: {self.selectedRect}")
            self.viewport().update()  # Trigger a repaint to show the rectangle
            # Trigger a repaint to show the rectangle

            for point in self.parent.parent.facialLandmarks.landmarks:
                if point.is_selected:
                    print(f"point {point.index} is selected")
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        elif self.isBboxMode:
            self.startPoint = None
        elif self.isSelectionMode and event.button() == Qt.LeftButton and self.isLeftMousePressed:
            # Check each keypoint if they are in the bbox and return a list of selected points
            self.startPoint = None
            self.isLeftMousePressed = False
            self.selectedRect = None
            self.viewport().update()
        else:
            super().mouseReleaseEvent(event)

    def selectPointsInRectangle(self):
        if self.selectedRect:
            print("Selecting points in rectangle")
            for point in self.parent.parent.facialLandmarks.landmarks:
                print(f"point {point.index} is visible: {point.show_in_canvas}")
                if self.selectedRect.contains(QPointF(point.x, point.y)) and point.show_in_canvas:
                    point.set_selection_status(True)
                else:
                    point.set_selection_status(False)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selectedRect:
            painter = QPainter(self.viewport())
            painter.setPen(QPen(QColor(0, 0, 255, 127), 2, Qt.DashLine))
            painter.setBrush(QColor(0, 0, 255, 50))
            painter.drawRect(self.mapFromScene(self.selectedRect).boundingRect())

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

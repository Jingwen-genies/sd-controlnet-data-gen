from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QGraphicsRectItem
from PyQt5.QtCore import Qt


class Bbox:
    def __init__(self, scene):
        self.scene = scene
        self.rectItem = None  # QGraphicsRectItem
        self.bbox = None

    def createOrUpdate(self, x1, y1, x2, y2):
        if self.rectItem is None:
            self.rectItem = QGraphicsRectItem(QRectF(x1, y1, x2-x1, y2-y1))
            self.rectItem.setPen(QPen(Qt.red, 2))
            self.scene.addItem(self.rectItem)
        else:
            self.rectItem.setRect(QRectF(x1, y1, x2-x1, y2-y1))
        self.bbox = (x1, y1, x2, y2)

    def remove(self):
        if self.rectItem:
            self.scene.removeItem(self.rectItem)
            self.rectItem = None
            self.bbox = None

    def draw(self):
        pen = QPen(self.color, 2)
        self.rectItem.setPen(pen)
        self.scene.addItem(self.rectItem)

    def hide(self):
        self.rectItem.hide()

    def show(self):
        self.rectItem.show()

    def setTransparency(self, alpha):
        self.rectItem.setOpacity(alpha)
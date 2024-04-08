from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtGui import  QFont


class Keypoint(QGraphicsEllipseItem):
    def __init__(self, poseObj,  x, y, visibility, radius=2, parent=None, index=0, color=Qt.yellow):
        super().__init__(x - radius, y - radius, 2 * radius, 2 * radius, parent)
        self.poseObj = poseObj
        self.x = x
        self.y = y
        self.initial_x = x
        self.initial_y = y
        self.visibility = visibility
        self.color = color
        self.radius = radius  # radius of the keypoint for different scales
        self.originalRadius = radius  # 保存原始半径
        # self.setFlag(QGraphicsEllipseItem.ItemIsMovable)  # 使landmark可拖动
        # self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges)  # 确保拖动时更新位置
        self.setAcceptHoverEvents(True)
        self.index_offset = 24
        self.show_in_canvas = True

        self.setFlags(QGraphicsEllipseItem.ItemIsMovable | QGraphicsEllipseItem.ItemSendsGeometryChanges)
        self.is_selected = False
        self.setBrush(QBrush(self.getColor()))


        # index
        self.index = index
        self.indexTextItem = QGraphicsTextItem(str(self.index + self.index_offset), self)
        self.indexTextItem.setDefaultTextColor(QColor(Qt.black))
        font = QFont()
        font.setPointSize(min(1, radius - 1))
        self.indexTextItem.setFont(font)
        self.updateTextPosition()

    def setVisibility(self, visibility):
        self.visibility = visibility
        self.is_selected = False

    def mousePressEvent(self, event):
        self.is_selected = not self.is_selected
        if self.is_selected:
            self.setBrush(QBrush(QColor(Qt.red)))
        else:
            self.setBrush(QBrush(self.color))
        super().mousePressEvent(event)

    def updateTextPosition(self):
        textRect = self.indexTextItem.boundingRect()
        self.indexTextItem.setPos(self.x - textRect.width() / 2, self.y - textRect.height() / 2)

    def showIndex(self):
        self.indexTextItem.setVisible(True)

    def hideIndex(self):
        self.indexTextItem.setVisible(False)

    def getColor(self):
        if self.is_selected:
            return QColor(Qt.red)
        elif self.visibility == 0.0:
            return QColor(Qt.gray)
        elif self.visibility == 1.0:
            return QColor(Qt.blue)
        else:
            return QColor(Qt.yellow)

    def updateColor(self, color):
        self.color = color
        self.color = self.getColor()
        self.setBrush(QBrush(self.color))

    def updatePosition(self):
        # self.setPos(self.x - self.radius, self.y - self.radius)
        self.setPos(self.x, self.y)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionHasChanged:
            print("keypoint position has changed to", value)
            scenePos = self.mapToScene(self.x, self.y)
            newPos = value
            self.x = self.initial_x + newPos.x()
            self.y = self.initial_y + newPos.y()
            self.poseObj.draw_connection()  # 通知FacialLandmarks更新连线
        return super().itemChange(change, value)

    def updateScale(self, scaleFactor):
        self.radius = self.originalRadius * scaleFactor
        self.setRect(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius)

    def setPosition(self, x, y):
        self.x = x
        self.y = y
        self.updatePosition()
        self.updateTextPosition()

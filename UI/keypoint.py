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
        self.initial_group_positions = None
        self.is_being_moved = False


        # index
        self.index = index
        self.indexTextItem = QGraphicsTextItem(str(self.index + self.index_offset), self)
        self.indexTextItem.setDefaultTextColor(QColor(Qt.black))
        font = QFont()
        font.setPointSize(max(1, radius - 1))
        self.indexTextItem.setFont(font)
        self.updateTextPosition()
        self.group, self.group_indices = self.poseObj.getGroup(self.index)

    def setVisibility(self, visibility):
        self.visibility = visibility
        self.is_selected = False

    def mousePressEvent(self, event):
        self.is_selected = not self.is_selected
        if self.is_selected:
            self.setBrush(QBrush(QColor(Qt.red)))
        else:
            self.setBrush(QBrush(self.color))
            # Store initial positions of the group keypoints when dragging starts
        self.is_being_moved = True  # Set the flag to true when the item is being moved
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.is_being_moved = False  # Reset the flag when the item is released
        super().mouseReleaseEvent(event)


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
        self.setPos(self.x - self.radius, self.y - self.radius)
        # self.setPos(self.x, self.y)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionHasChanged:
            print(f"index: {self.index} calling itemChange, with change: {change}, value: {value}")
            print(f"initial x: {self.initial_x}, initial y: {self.initial_y}")
            scenePos = self.mapToScene(self.x, self.y)
            newPos = value
            prev_x = self.x
            prev_y = self.y
            self.x = self.initial_x + newPos.x()
            self.y = self.initial_y + newPos.y()

            print(f"prev_x: {prev_x}, prev_y: {prev_y}")
            print(f"new x: {self.x}, new y: {self.y}")
            print(f"scenePos: {scenePos}")
            print(f"newPos: {newPos}")
            if self.poseObj.view.shiftPressed and self.is_being_moved:
                print(" ")
                print("Dealing with group keypoints")
                delta_x = self.x - prev_x
                delta_y = self.y - prev_y
                print(f"delta_x: {delta_x}, delta_y: {delta_y}")
                print(f"point is coming from group: {self.group} with indices: {self.group_indices}")
                for idx in self.group_indices:
                    if idx != self.index:  # Skip the current keypoint as it's already updated
                        print("updating keypoint", idx)
                        print("previous x:", self.poseObj.landmarks[idx].x, "previous y:", self.poseObj.landmarks[idx].y)
                        self.poseObj.landmarks[idx].x += delta_x
                        self.poseObj.landmarks[idx].y += delta_y
                        print("new x:", self.poseObj.landmarks[idx].x, "new y:", self.poseObj.landmarks[idx].y)
                        # self.poseObj.landmarks[idx].updatePosition()
                        print(f"updated keypoint {idx} to x: {self.poseObj.landmarks[idx].x}, y: {self.poseObj.landmarks[idx].y}")
                        print(" ")
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
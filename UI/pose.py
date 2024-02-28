from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtGui import QPen, QFont
from typing import List
# TODO: figure out how does the circle been drawed, get the center and radius of the circle, draw text at the center of the circle

class keypoint(QGraphicsEllipseItem):
    def __init__(self, poseObj, x, y, visibility, radius=3, parent=None, color=Qt.yellow):
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
        self.setBrush(QBrush(self.getColor()))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable)  # 使landmark可拖动
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges)  # 确保拖动时更新位置
        self.setAcceptHoverEvents(True)

    def getColor(self):
        if self.visibility == 0:
            return QColor(Qt.red)
        elif self.visibility == 1:
            return self.color
        else:
            return QColor(Qt.blue)

    def updatePosition(self):
        # self.setPos(self.x - self.radius, self.y - self.radius)
        self.setPos(self.x, self.y)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionHasChanged:
            # 当landmark位置改变时，可以在这里更新它的u, v坐标
            scenePos = self.mapToScene(self.x, self.y)
            # newPos = value
            newPos = value
            self.x = self.initial_x + newPos.x()
            self.y = self.initial_y + newPos.y()
            self.poseObj.draw_connection()  # 通知FacialLandmarks更新连线
        return super().itemChange(change, value)

    def updateScale(self, scaleFactor):
        # 根据缩放因子调整landmark的大小
        self.radius = self.originalRadius * scaleFactor
        self.setRect(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius)


class FacialLandmarks:
    def __init__(self, scene, landmarks: List[List[int]], sceneWidth=512, sceneHeight=512, color=Qt.yellow):
        """
        Args:
            scene: QtGraphicsScene
            landmarks: list of keypoints in 2D ,they are simply just lists of [x, y, visibility]
            note that the landmarks here are values in the range of [0, 1],
            we should scale them by canvas size when drawing them
        """
        self.color = color
        self.scene = scene
        self.sceneWidth = sceneWidth
        self.sceneHeight = sceneHeight
        self.landmarks = [keypoint(self, u * sceneWidth, v * sceneHeight, visibility, color=color) for u, v, visibility in landmarks]
        self.connections = {
            "jaw": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "left_eyebrow": [17, 18, 19, 20, 21],
            "right_eyebrow": [22, 23, 24, 25, 26],
            "nose_bridge": [27, 28, 29, 30],
            "nose_tip": [31, 32, 33, 34, 35],
            "left_eye": [36, 37, 38, 39, 40, 41, 36],
            "right_eye": [42, 43, 44, 45, 46, 47, 42],
            "upper_lip": [48, 49, 50, 51, 52, 53, 54],
            "lower_lip": [55, 54, 55, 56, 57, 58, 59],
            "upper_inner_lip": [60, 61, 62, 63, 64],
            "lower_inner_lip": [65, 66, 67]
        }
        self.lines = []


    def draw_connection(self):
        # 移除旧的连线
        for line in self.lines:
            self.scene.removeItem(line)
        self.lines.clear()  # 清空连线列表

        # 绘制新的连线
        pen = QPen(self.color, 2)  # 定义线条样式
        for part, indices in self.connections.items():
            for i in range(len(indices) - 1):
                start_idx, end_idx = indices[i], indices[i + 1]
                start, end = self.landmarks[start_idx], self.landmarks[end_idx]
                if start.visibility == 1 and end.visibility == 1:
                    line = QGraphicsLineItem(start.x + start.radius, start.y + start.radius, end.x + end.radius,
                                             end.y + end.radius)
                    line.setPen(pen)
                    self.scene.addItem(line)
                    self.lines.append(line)  # 添加新的连线到列表

    def draw(self):
        for kp in self.landmarks:
            self.scene.removeItem(kp)
        for line in self.lines:
            self.scene.removeItem(line)
        # 添加关键点到场景
        for i, kp in enumerate(self.landmarks):
            self.scene.addItem(kp)
            # 创建一个文本项显示关键点索引
            textItem = QGraphicsTextItem(str(i))
            # draw the text at the center of the keypoint circle
            textItem.setPos(kp.x - kp.radius / 2, kp.y - kp.radius / 2)

            textItem.setDefaultTextColor(QColor(Qt.white))
            # 设置字体大小
            font = QFont()
            font.setPointSize(5)  # 设置字体大小为12
            textItem.setFont(font)
            self.scene.addItem(textItem)
        self.draw_connection()


    def getLandmarks(self):
        import numpy as np
        points = np.array([[kp.x / self.sceneWidth, kp.y / self.sceneHeight, kp.visibility] for kp in self.landmarks]).reshape(-1)
        return points.tolist()

    def hide(self):
        for kp in self.landmarks:
            kp.hide()
        for line in self.lines:
            line.hide()

    def show(self):
        for kp in self.landmarks:
            kp.show()
        for line in self.lines:
            line.show()

    def setTransparency(self, alpha):
        for kp in self.landmarks:
            kp.setOpacity(alpha)
        for line in self.lines:
            line.setOpacity(alpha)

from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsRectItem
from PyQt5.QtCore import Qt, QRectF, QTimer, QPointF
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter
from PyQt5.QtGui import QPen, QFont
from typing import List
import numpy as np
import cv2

from keypoint import Keypoint


class FacialLandmarks:
    def __init__(self, scene, view, landmarks: List[List[int]], sceneWidth=512, sceneHeight=512, color=Qt.yellow):
        """
        Args:
            scene: QtGraphicsScene
            landmarks: list of keypoints in 2D ,they are simply just lists of [x, y, visibility]
            note that the landmarks here are values in the range of [0, 1],
            we should scale them by canvas size when drawing them
        """
        self.color = color
        self.scene = scene
        self.view = view
        self.sceneWidth = sceneWidth
        self.sceneHeight = sceneHeight
        self.landmarks = [Keypoint(self, u * sceneWidth, v * sceneHeight, visibility, index=index, color=color) for index, (u, v, visibility) in enumerate(landmarks)]
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
        self.isVisiable = True

    def clear(self):
        """Clear all the landmark related element from the scene"""
        for kp in self.landmarks:
            if kp.scene() == self.scene:
                self.scene.removeItem(kp)
        for line in self.lines:
            if line.scene() == self.scene:
                self.scene.removeItem(line)

    def setSceneVisibility(self, visibility, index_group):
        """
        Show, hide landmark groups
        Args:
            visibility: true: show, false: hide
            index_group: string, must be chosen from ["eyes", "nose", "mouth", "outline"]
        """
        # Define the indices for each group
        group_indices = {
            "eyes": list(range(17, 27)) + list(range(36, 48)),  # Combines the two ranges for eyes
            "nose": list(range(27, 36)),
            "mouth": list(range(48, 68)),  # Adjusted assuming 68 as an example endpoint
            "outline": list(range(17))  # This selects indices from 0 to 16
        }
        if len(self.landmarks) > 68:
            group_indices["eyes"] += [68, 69]

        # Check if the provided index group is valid
        if index_group in group_indices:
            # Iterate over the indices for the specified group
            for idx in group_indices[index_group]:
                # Ensure the index is within the bounds of the landmarks list
                if idx < len(self.landmarks):
                    self.landmarks[idx].show_in_canvas = visibility
        else:
            print(f"Unknown index group: {index_group}")

    def addLeftPupil(self):
        # calculating the pupils
        left_eye = self.landmarks[42: 48]
        left_eye = [[kp.x, kp.y] for kp in left_eye]
        left_eye_center = [sum([x[0] for x in left_eye]) / 6, sum([x[1] for x in left_eye]) / 6]
        left_pupil = Keypoint(self, left_eye_center[0], left_eye_center[1], 2, index=68, color=self.color)
        self.landmarks.append(left_pupil)

    def addRightPupil(self):
        right_eye = self.landmarks[36: 42]
        right_eye = [[kp.x, kp.y] for kp in right_eye]
        right_eye_center = [sum([x[0] for x in right_eye]) / 6, sum([x[1] for x in right_eye]) / 6]
        right_pupil = Keypoint(self, right_eye_center[0], right_eye_center[1], 2, index=69, color=self.color)
        self.landmarks.append(right_pupil)

    def removeLeftPupil(self):
        if len(self.landmarks) > 68:
            self.landmarks[68].setIvisible()

    def removeRightPupil(self):
        if len(self.landmarks) > 69:
            self.landmarks[69].setIvisible()

    def draw_connection(self):
        # 移除旧的连线
        for line in self.lines:
            if line.scene() == self.scene:
                self.scene.removeItem(line)
        self.lines.clear()

        pen = QPen(self.color, 2)
        for part, indices in self.connections.items():
            for i in range(len(indices) - 1):
                start_idx, end_idx = indices[i], indices[i + 1]
                start, end = self.landmarks[start_idx], self.landmarks[end_idx]
                # line = QGraphicsLineItem(start.x + start.radius, start.y + start.radius, end.x + end.radius,
                #                          end.y + end.radius)
                line = QGraphicsLineItem(start.x, start.y, end.x, end.y)
                line.setPen(pen)
                line.setOpacity(0.5)
                self.scene.addItem(line)
                self.lines.append(line)
                if not start.show_in_canvas or not end.show_in_canvas:
                    line.hide()

    def draw(self):
        for kp in self.landmarks:
            if kp.scene() == self.scene:
                self.scene.removeItem(kp)
        for line in self.lines:
            if line.scene() == self.scene:
                self.scene.removeItem(line)
        # self.lines.clear()
        for i, kp in enumerate(self.landmarks):
            if kp.show_in_canvas:
                kp.updateColor(self.color)
                self.scene.addItem(kp)
            # self.scene.addItem(kp.indexTextItem)
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
        self.isVisiable = False

    def show(self):
        for kp in self.landmarks:
            kp.show()
        for line in self.lines:
            line.show()
        self.isVisiable = True

    def showIndex(self):
        for kp in self.landmarks:
            kp.showIndex()

    def hideIndex(self):
        for kp in self.landmarks:
            kp.hideIndex()

    def setTransparency(self, alpha):
        for kp in self.landmarks:
            kp.setOpacity(alpha)
        for line in self.lines:
            line.setOpacity(alpha)

    def saveImage(self, tgt_path):
        print(f"saving image: {self.sceneWidth}x{self.sceneHeight} to {tgt_path}")
        # plot the landmarks to a 512x512 image, no face, just the points, black background white point
        # use cv2 or Image to draw the landmarks
        # create a empty numpy array
        img = np.zeros((self.sceneHeight, self.sceneWidth, 3), dtype=np.uint8)
        # put the point to the image, the point is white
        for kp in self.landmarks:
            if kp.visibility == 1:
                cv2.circle(img, (int(kp.x), int(kp.y)), 1, (255, 255, 255), -1)
        # save the image
        cv2.imwrite(tgt_path, img)

    def setVisibility(self, visibility):
        for kp in self.landmarks:
            if kp.is_selected:
                kp.setVisibility(visibility)

    def selectAll(self):
        for kp in self.landmarks:
            kp.is_selected = True
            # kp.setBrush(QBrush(QColor(Qt.red)))

    def getGroup(self, index):
        # return the group of the index where the index is in based on self.connections
        for key, value in self.connections.items():
            if index in value:
                return key, value

    def moveGroup(self, index, offset):
        # move the group of the index together with index
        print(f"landmark: shift pressed: {self.view.shiftPressed}")
        if self.view.shiftPressed:
            group, indices = self.getGroup(index)
            for idx in indices:
                if idx != index:
                    self.landmarks[idx].x += offset[0]
                    self.landmarks[idx].y += offset[1]
            self.draw()
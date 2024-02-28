import sys
import json
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QMainWindow,
    QGraphicsView,
    QGraphicsScene,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QGraphicsPixmapItem,
    QSlider,
)

from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import QDir, Qt, pyqtSignal
import dataclasses
from typing import List

from utils import read_openpose, read_json, write_json
from pose import FacialLandmarks
# from client import runtime_sm_client, create_json_request, get_landmarks_from_response

@dataclasses.dataclass
class csvRow:
    image_path: str
    landmark: str
    is_kept: bool


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose annotation tool")
        self.setGeometry(100, 100, 800, 600)

        # Create the main widget and layout
        self.mainWidget = QWidget()
        self.mainLayout = QVBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)

        # Create and set the graphics view and scene
        self.graphicsView = CustomGraphicsView()
        self.graphicsScene = QGraphicsScene()
        self.graphicsView.setScene(self.graphicsScene)
        self.mainLayout.addWidget(self.graphicsView)

        # Add buttons
        self.prevButton = QPushButton("Previous")
        self.nextButton = QPushButton("Next")
        self.discardButton = QPushButton("Discard")
        self.saveButton = QPushButton("Save")
        self.hideButton = QPushButton("Hide Landmarks")
        # sliding scale for transparency
        self.transparencySlider = QSlider(Qt.Horizontal)
        self.transparencySlider.setRange(0, 100)
        self.transparencySlider.setValue(100)
        self.detectionButton = QPushButton("Run Facial Landmark Detection")
        self.replaceButton = QPushButton("Replace Landmark")
        self.addBboxButton = QPushButton("Add Bounding Box")


        # Connect buttons to functions
        self.prevButton.clicked.connect(self.prev_image)
        self.nextButton.clicked.connect(self.next_image)
        self.discardButton.clicked.connect(self.discard_image)
        self.saveButton.clicked.connect(self.save_landmarks)
        self.hideButton.clicked.connect(self.hide_landmarks)
        self.transparencySlider.valueChanged.connect(self.set_transparency)
        self.detectionButton.clicked.connect(self.run_detection)
        self.replaceButton.clicked.connect(self.replace_landmark)
        self.addBboxButton.clicked.connect(self.add_bounding_box)


        # Add buttons to layout
        self.mainLayout.addWidget(self.prevButton)
        self.mainLayout.addWidget(self.nextButton)
        self.mainLayout.addWidget(self.discardButton)
        self.mainLayout.addWidget(self.saveButton)
        self.mainLayout.addWidget(self.hideButton)
        self.mainLayout.addWidget(self.transparencySlider)
        self.mainLayout.addWidget(self.detectionButton)
        self.mainLayout.addWidget(self.replaceButton)
        self.mainLayout.addWidget(self.addBboxButton)


        self.csvData_list:List[csvRow] = []
        self.currentIndex = -1
        self.facialLandmarks = None
        self.json = None
        self.hide = False
        self.detectedLandmarks = None

    def add_bounding_box(self):
        # TODO: add bounding box
        pass

    def run_detection(self):
        image_paths = [self.csvData_list[self.currentIndex].image_path]
        payload = create_json_request(paths=image_paths, radius=3, show_kpt_idx=True)
        endpoint_name = "facial-landmark-app-v2"
        results = runtime_sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        detected_landmarks = get_landmarks_from_response(results)
        self.detectedLandmarks = FacialLandmarks(self.graphicsScene, detected_landmarks, sceneWidth=512, sceneHeight=512, color=Qt.green)
        # TODO: plot the landmarks on the image?

    def replace_landmark(self):
        self.facialLandmarks = self.detectedLandmarks
        self.facialLandmarks.draw()

    def set_transparency(self, value):
        alpha = value / 100.0
        # set the transparency of the landmarks according to the slider value
        if self.facialLandmarks:
            self.facialLandmarks.setTransparency(alpha)

    def hide_landmarks(self):
        # click once to hide, click again to show
        if self.hide:
            self.facialLandmarks.hide()
            self.hide = False
        else:
            self.facialLandmarks.show()
            self.hide = True

    def discard_image(self):
        self.csvData_list[self.currentIndex].is_kept = False

    def load_csv(self):
        # csv_file = QFileDialog.getOpenFileName(self, 'Open CSV', QDir.currentPath(), 'CSV Files (*.csv)')
        # if csv_file[0]:
        #     with open(csv_file[0], 'r') as file:
        #         for line in file:
        #             if line.startswith('image'):
        #                 continue
        #             data = line.split(',')
        #             self.csvData_list.append(csvRow(data[0], data[1], data[2]))
        #     self.currentIndex = 0
        #     self.update_curr_img_pose()
        # else:
        #     print("No file selected")
        csv_file = r"C:\Users\Jingwen\Documents\projs\stable-diffusion-webui\avatar_generation\outputs\synthetic_data_info.csv"
        with open(csv_file, 'r') as file:
            for line in file:
                if line.startswith('image'):
                    continue
                data = line.split(',')
                self.csvData_list.append(csvRow(data[0], data[1], data[2]))
        self.currentIndex = 0
        self.update_curr_img_pose()

    def update_curr_img_pose(self):
        if self.currentIndex < 0 or self.currentIndex >= len(self.csvData_list):
            return
        self.load_image()
        self.load_json()

    def prev_image(self):
        self.currentIndex -= 1
        self.update_curr_img_pose()

    def next_image(self):
        self.currentIndex += 1
        self.update_curr_img_pose()

    def load_image(self):
        if self.currentIndex < 0 or self.currentIndex >= len(self.csvData_list):
            return

        # 从文件加载图像
        pixmap = QPixmap(self.csvData_list[self.currentIndex].image_path)

        # 调整图像大小到512x512，同时保持宽高比
        scaled_pixmap = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmapItem = QGraphicsPixmapItem(scaled_pixmap)

        # 将调整大小后的图像设置到QLabel上
        self.graphicsScene.addItem(pixmapItem)

        # 居中显示图像（如果你想这样做的话）
        self.graphicsView.fitInView(pixmapItem, Qt.KeepAspectRatio)

    def load_json(self):
        if self.currentIndex < 0 or self.currentIndex >= len(self.csvData_list):
            return
        landmark_path = self.csvData_list[self.currentIndex].landmark
        print(landmark_path)
        self.json_dictionary = read_json(landmark_path)
        landmarks = read_openpose(landmark_path)
        scene_width = 512
        scene_height = 512
        if landmarks is not None:
            # Note that here the landmarks are scaled as uv coordinates (0-1)
            # 假设read_openpose返回的是一个包含(x, y, visibility)的列表
            self.facialLandmarks = FacialLandmarks(self.graphicsScene, landmarks, sceneWidth=scene_width, sceneHeight=scene_height )
            self.facialLandmarks.draw()
        else:
            print("No landmark found")

    def updateFacialLandmarks(self, scaleFactor):
        currentImageWidth = self.graphicsView.getCurrentImageWidth()  # 需要自己实现获取当前图片宽度的方法
        currentImageHeight = self.graphicsView.getCurrentImageHeight()  # 需要自己实现获取当前图片高度的方法
        self.facialLandmarks.updateKeypoints(scaleFactor, currentImageWidth, currentImageHeight)

    def save_landmarks(self):
        landmarks = self.facialLandmarks.getLandmarks()
        self.json_dictionary["people"][0]['face_keypoints_2d'] = landmarks
        # print(self.json_dictionary)
        json_path = self.csvData_list[self.currentIndex].landmark
        print("json path:", json_path)
        # Check if the JSON file path is a valid string
        if isinstance(json_path, str):
            # Attempt to write the JSON dictionary to the specified file
            try:
                write_json(self.json_dictionary, json_path)
                print("Landmarks saved to JSON file successfully.")
            except Exception as e:
                print("Error:", e)
        else:
            print("Error: Invalid JSON file path.")


class CustomGraphicsView(QGraphicsView):
    zoomChanged = pyqtSignal(float)  # 定义一个新的信号，传递缩放因子

    def __init__(self, parent=None):
        super(CustomGraphicsView, self).__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        # 设置缩放的初始级别
        self.scaleFactor = 1.0

    def wheelEvent(self, event):
        # 每次缩放的比例因子
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        # 设置缩放的限制条件
        minFactor = 0.2
        maxFactor = 5.0

        # 获取滚轮的滚动方向来决定是放大还是缩小
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor

        # 计算新的缩放级别并应用限制
        newScaleFactor = self.scaleFactor * zoomFactor
        if newScaleFactor < minFactor or newScaleFactor > maxFactor:
            return

        self.scale(zoomFactor, zoomFactor)
        self.scaleFactor = newScaleFactor

    def getViewWidth(self):
        return self.viewport().width()

    def getViewHeight(self):
        return self.viewport().height()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MainWindow()
    viewer.show()
    viewer.load_csv()
    sys.exit(app.exec_())
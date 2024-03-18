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
    QHBoxLayout,
    QWidget,
    QGraphicsPixmapItem,
    QSlider,
)

from PyQt5.QtGui import QPixmap, QPainter, QIcon
from PyQt5.QtCore import QDir, Qt, pyqtSignal, QTimer
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
        self.setGeometry(100, 100, 1500, 1024)  # 确保主窗口足够大
        self.hButtonSpace = 10  # 控制按钮之间的间距
        self.hStretch = 1  # 控制按钮之间的弹性空间
        self.csvData_list:List[csvRow] = []
        self.currentIndex = -1
        self.facialLandmarks = None
        self.json = None
        self.hide = False
        self.detectedLandmarks = None


        # 创建中央widget和布局
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainVerticalLayout = QVBoxLayout(centralWidget)

        topHorizontalLayout = QHBoxLayout()
        # 创建并添加控制面板到主布局
        self.leftControlPanel = LeftControlPanel(self)

        # 中央图片显示区域
        self.imageViewer = ImageViewer()  # 自定义的ImageViewer类
        self.imageViewer.requestPreviousImage.connect(self.prev_image)
        self.imageViewer.requestNextImage.connect(self.next_image)

        self.rightControlPanel = RightControlPanel(self)

        topHorizontalLayout.addWidget(self.leftControlPanel)
        topHorizontalLayout.addWidget(self.imageViewer, 1)  # ImageViewer占据多余空间
        topHorizontalLayout.addWidget(self.rightControlPanel)

        # 将水平布局添加到垂直布局
        mainVerticalLayout.addLayout(topHorizontalLayout)

        self.bottomControlPanel = BottomControlPanel(self)
        mainVerticalLayout.addWidget(self.bottomControlPanel)


    def set_bbox_transparency(self, value):
        pass

    def add_bounding_box(self):
        # TODO: add bounding box
        pass

    def save_bounding_box(self):
        pass

    def run_bounding_box_detection(self):
        pass

    def run_facial_landmark_detection(self):
        pass
        # image_paths = [self.csvData_list[self.currentIndex].image_path]
        # payload = create_json_request(paths=image_paths, radius=3, show_kpt_idx=True)
        # endpoint_name = "facial-landmark-app-v2"
        # results = runtime_sm_client.invoke_endpoint(
        #     EndpointName=endpoint_name,
        #     ContentType="application/json",
        #     Body=payload,
        # )
        # detected_landmarks = get_landmarks_from_response(results)
        # self.detectedLandmarks = FacialLandmarks(self.graphicsScene, detected_landmarks, sceneWidth=512, sceneHeight=512, color=Qt.green)
        # # TODO: plot the landmarks on the image?


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
        self.imageViewer.load_image(self.csvData_list, self.currentIndex)
        self.load_json()

    def prev_image(self):
        self.currentIndex -= 1
        self.update_curr_img_pose()

    def next_image(self):
        self.currentIndex += 1
        self.update_curr_img_pose()

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
            self.facialLandmarks = FacialLandmarks(self.imageViewer.graphicsScene, landmarks, sceneWidth=scene_width, sceneHeight=scene_height )
            self.facialLandmarks.draw()
        else:
            print("No landmark found")

    def updateFacialLandmarks(self, scaleFactor):
        currentImageWidth = self.imageViewer.graphicsView.getCurrentImageWidth()  # 需要自己实现获取当前图片宽度的方法
        currentImageHeight = self.imageViewer.getCurrentImageHeight()  # 需要自己实现获取当前图片高度的方法
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
        # self.userHasZoomed = False  # 跟踪用户是否已经进行了缩放
        self.shouldFitInView = True  # 初始时允许fitInView
    def resizeEvent(self, event):
        # 在窗口大小改变时调整视图
        if self.shouldFitInView and self.scene():
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def wheelEvent(self, event):
        self.shouldFitInView = False
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


class ImageViewer(QWidget):
    """
    center image and the prev, next button 中央图片显示区域
    """
    requestPreviousImage = pyqtSignal()  # 请求显示上一个图片的信号
    requestNextImage = pyqtSignal()  # 请求显示下一个图片的信号
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)

        # 使用CustomGraphicsView作为图片显示区域
        self.graphicsView = CustomGraphicsView()
        self.graphicsScene = QGraphicsScene(self)


        self.graphicsView.setScene(self.graphicsScene)

        self.prevButton = QPushButton("<", self)
        self.nextButton = QPushButton(">", self)
        # 设置导航按钮样式
        self.prevButton.setStyleSheet("background-color: darkgray; border: none;")
        self.nextButton.setStyleSheet("background-color: darkgray; border: none;")
        # 设置导航按钮半透明
        self.prevButton.setWindowOpacity(0.5)
        self.nextButton.setWindowOpacity(0.5)
        self.prevButton.setMinimumSize(30, 40)  # 设置最小宽度为60，最小高度为40
        self.nextButton.setMinimumSize(30, 40)

        # 创建水平布局并添加组件
        layout = QHBoxLayout(self)
        layout.addWidget(self.prevButton)
        layout.addWidget(self.graphicsView, 1)  # 加1使graphicsView可以扩展填充剩余空间
        layout.addWidget(self.nextButton)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除边距以使按钮更靠近边缘

        self.prevButton.clicked.connect(self.requestPreviousImage.emit)
        self.nextButton.clicked.connect(self.requestNextImage.emit)
        print("initializing the image viewer")
        print("adjust botton postion")
        print(f"size: {self.size()}")
        print(f"height: {self.size().height()}")
        # 使用QTimer来确保在布局稳定后再调整按钮位置
        QTimer.singleShot(0, self.adjustButtonPositions)

    def load_image(self, csvData_list, currentIndex):
        # 从文件加载图像
        pixmap = QPixmap(csvData_list[currentIndex].image_path)

        # 调整图像大小到512x512，同时保持宽高比
        scaled_pixmap = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmapItem = QGraphicsPixmapItem(scaled_pixmap)

        # 将调整大小后的图像设置到QLabel上
        self.graphicsScene.addItem(pixmapItem)

        # 居中显示图像（如果你想这样做的话）
        self.graphicsView.fitInView(pixmapItem, Qt.KeepAspectRatio)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjustButtonPositions()

    def adjustButtonPositions(self):
        # 假设initialViewHeight是ImageViewer的初始高度
        initialViewHeight = self.height()

        btnY = initialViewHeight // 2 - self.prevButton.height() // 2
        print(f"adjusting height to {btnY}")
        self.prevButton.move(10, btnY)
        self.nextButton.move(self.width() - self.nextButton.width() - 10, btnY)


class LeftControlPanel(QWidget):
    def __init__(self, mainWindow, parent=None):
        super(LeftControlPanel, self).__init__(parent)
        self.mainWindow = mainWindow

        layout = QVBoxLayout()
        self.setLayout(layout)

        # sliding scale for transparency
        self.transparencySlider = QSlider(Qt.Horizontal)
        self.transparencySlider.setRange(0, 100)
        self.transparencySlider.setValue(100)
        self.runBoundingBoxDetectionButton = QPushButton("Run Bounding Box Detection")
        self.addBoundingBoxButton = QPushButton("Add Bounding Box")
        buttons = [self.transparencySlider, self.runBoundingBoxDetectionButton, self.addBoundingBoxButton]
        maxWidth = 200  # 控制元素的最大宽度
        for button in buttons:
            layout.addWidget(button)
            button.setFixedWidth(maxWidth)

        self.setFixedWidth(maxWidth + 20)


        # 设置布局的间距和边距
        layout.setSpacing(mainWindow.hButtonSpace)  # 设置控件之间的间距
        layout.setContentsMargins(10, 10, 10, 10)  # 设置布局的边距
        # 在添加了所有控件之后添加弹性空间
        layout.addStretch(mainWindow.hStretch)

        # 连接信号到槽
        self.transparencySlider.valueChanged.connect(self.mainWindow.set_bbox_transparency)
        self.runBoundingBoxDetectionButton.clicked.connect(self.mainWindow.run_bounding_box_detection)
        self.addBoundingBoxButton.clicked.connect(self.mainWindow.add_bounding_box)


class RightControlPanel(QWidget):
    def __init__(self, mainWindow, parent=None):
        super(RightControlPanel, self).__init__(parent)
        self.mainWindow = mainWindow

        layout = QVBoxLayout()
        self.setLayout(layout)

        # sliding scale for transparency
        self.transparencySlider = QSlider(Qt.Horizontal)
        self.transparencySlider.setRange(0, 100)
        self.transparencySlider.setValue(100)
        self.hideLandmarkIndexButton = QPushButton("Hide Landmark Index")
        self.runFacialLandmarkDetectionButton = QPushButton("Run Facial Landmark Detection")
        buttons = [self.transparencySlider, self.hideLandmarkIndexButton, self.runFacialLandmarkDetectionButton]
        maxWidth = 200
        for button in buttons:
            layout.addWidget(button)
            button.setFixedWidth(maxWidth)

        self.setFixedWidth(maxWidth + 20)


        # 设置布局的间距和边距
        layout.setSpacing(mainWindow.hButtonSpace)  # 设置控件之间的间距
        layout.setContentsMargins(10, 10, 10, 10)  # 设置布局的边距
        # 在添加了所有控件之后添加弹性空间
        layout.addStretch(mainWindow.hStretch)

        # 连接信号到槽
        self.transparencySlider.valueChanged.connect(self.mainWindow.set_transparency)
        self.runFacialLandmarkDetectionButton.clicked.connect(self.mainWindow.run_facial_landmark_detection)
        self.hideLandmarkIndexButton.clicked.connect(self.mainWindow.hide_landmarks)


class BottomControlPanel(QWidget):
    discardSignal = pyqtSignal()  # 丢弃图片的信号
    saveSignal = pyqtSignal()  # 保存标记的信号
    replaceSignal = pyqtSignal()  # 替换标记的信号

    def __init__(self, mainWindow, parent=None):
        super(BottomControlPanel, self).__init__(parent)
        self.mainWindow = mainWindow
        # 创建垂直布局
        layout = QHBoxLayout()
        # 设置布局
        self.setLayout(layout)
        layout.addStretch()

        # 创建按钮
        self.discardButton = QPushButton("Discard Image")
        self.saveButton = QPushButton("Save")
        self.replaceButton = QPushButton("Replace Landmark")
        buttons = [self.discardButton, self.saveButton, self.replaceButton]

        # 添加按钮到布局
        maxWidth = 200  # 控制元素的最大宽度
        for button in buttons:
            layout.addWidget(button)
            button.setFixedWidth(maxWidth)
        layout.addStretch()


        # 连接按钮的信号到槽
        self.discardButton.clicked.connect(self.discardSignal.emit)
        self.saveButton.clicked.connect(self.saveSignal.emit)
        self.replaceButton.clicked.connect(self.replaceSignal.emit)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MainWindow()
    viewer.show()
    viewer.load_csv()
    sys.exit(app.exec_())
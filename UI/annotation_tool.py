import sys
from pathlib import Path
import boto3
import logging
from PyQt5.QtWidgets import (
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
    QCheckBox,
    QAction,
    QFrame,
    QLabel, QLineEdit, QMessageBox
)

from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import dataclasses
from typing import List
import os
import rootutils
import warnings
import csv
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="pytorch_lightning.utilities.distributed",
                        message=".*rank_zero_only has been deprecated.*")
project_root = rootutils.setup_root(search_from=__file__, indicator=".project-root", dotenv=True, pythonpath=True,
                                    cwd=False)
os.chdir(project_root)
from avatar_generation.support.utils import read_openpose, read_json, write_json, generate_csv
from avatar_generation.UI.facial_landmarks import FacialLandmarks
from avatar_generation.UI. keypoint import Keypoint
from avatar_generation.UI.bbox import Bbox
# from avatar_generation.UI.keypoint import Keypoint
from avatar_generation.UI.client import create_json_request, get_landmarks_from_response


@dataclasses.dataclass
class csvRow:
    image_path: str
    landmark: str
    is_kept: bool


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose annotation tool")
        self.runtime_sm_client = boto3.client(service_name="sagemaker-runtime")
        self.setGeometry(0, 0, 1280, 720)  # 确保主窗口足够大
        self.hButtonSpace = 10  # 控制按钮之间的间距
        self.hStretch = 1  # 控制按钮之间的弹性空间
        self.csvData_list: List[csvRow] = []
        self.currentIndex = -1
        self.facialLandmarks = None
        self.detectedLandmarks = None
        self.bbox = None
        self.json_dictionary = None
        self.labeling_control_image = False
        self.csv_path = ""
        self.landmark_template = None



        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')

        loadControlImageAction = QAction('&Label Control Images', self)
        loadControlImageAction.triggered.connect(self.load_control_image_csv)

        loadTraningImageAction = QAction('&Label Training Images', self)
        loadTraningImageAction.triggered.connect(self.load_training_image_csv)

        fileMenu.addAction(loadControlImageAction)
        fileMenu.addAction(loadTraningImageAction)

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        main_layout = QHBoxLayout(centralWidget)

        self.imageViewer = ImageViewer()

        self.imageViewer.requestPreviousImage.connect(self.prev_image)
        self.imageViewer.requestNextImage.connect(self.next_image)
        self.imageViewer.graphicsView.bboxModeChanged.connect(self.updateBboxSwitch)


        self.leftControlPanel = ControlPanel(self)

        main_layout.addWidget(self.imageViewer, 1)
        main_layout.addWidget(self.leftControlPanel)

        main_layout.addLayout(main_layout)

        # set default csv as training csv (default labeling training data)
        self.load_training_image_csv()

    def updateBboxSwitch(self, isBboxMode):
        self.leftControlPanel.addBboxSwitch.setChecked(isBboxMode)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left or event.key() == Qt.Key_A:
            self.prev_image()
        elif event.key() == Qt.Key_Right or event.key() == Qt.Key_D:
            self.next_image()
        elif event.key() == Qt.Key_E:
            if self.facialLandmarks:
                self.hide_landmarks()
            self.run_facial_landmark_detection()
            self.replace_landmark()
            self.add_left_pupil()
            self.add_right_pupil()
        # Enter key runs jump to index and numpad enter key runs save landmarks
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.jump_to_index()
        # save the landmarks using control + s
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.save_everything()
        elif event.key() == Qt.Key_V:
            self.setVisibility(2)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.write_data_to_csv()
        event.accept()

    def updateBboxModeCheckbox(self, isBboxMode):
        self.leftControlPanel.addBboxSwitch.setChecked(isBboxMode)

    def write_data_to_csv(self):
        # write the data to self.csvData_list to the csv file
        print("Writing data to csv file")
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["image", "landmark", "is_kept"])
            for row in self.csvData_list:
                is_kept_str = str(row.is_kept).strip().strip('"')
                writer.writerow([row.image_path, row.landmark, is_kept_str])

    def resetScene(self):
        # reset the QGraphicsScene and Qgraphiitems
        print("Resetting scene")
        self.imageViewer.graphicsScene.clear()
        self.imageViewer.graphicsView.scaleFactor = 1.0
        self.facialLandmarks = None
        self.detectedLandmarks = None
        self.bbox = None
        self.json_dictionary = None
        print(f"reset scaled factor to: {self.imageViewer.graphicsView.scaleFactor}")

    def toggleBboxMode(self, enabled):
        print("toggleBboxMode: ", enabled)
        self.imageViewer.graphicsView.isBboxMode = enabled

    def saveBbox(self):
        try:
            currentBbox = self.imageViewer.graphicsView.currentBbox
            if currentBbox:
                # Assuming you have a method or logic to serialize and save the bbox
                # For demonstration, simply printing the bbox coordinates
                print(f"Saving BBox: {currentBbox.bbox} to self.bbox")
                # save the bbox with format (x1, y1, x2, y2) as a tuple to self.bbox, where x1, y1 is the topleft point
                # and x2,y2 is the bottomright point
                self.bbox = currentBbox
                # Reset or clear the bbox after saving as needed
                self.imageViewer.graphicsView.currentBbox = None
        except Exception as e:
            print("Error saving bbox:", e)

    def run_facial_landmark_detection(self):
        print("Running facial landmark detection")
        image_paths = [self.csvData_list[self.currentIndex].image_path]
        if self.bbox:
            print("Running Facial Landmark Detection using self.bbox")
            x1, y1, x2, y2 = self.bbox.bbox
            bboxes = [[x1, y1, x2, y2]]
            payload = create_json_request(image_paths, bounding_box=bboxes)
        else:
            payload = create_json_request(paths=image_paths, radius=3, show_kpt_idx=True)
        endpoint_name = "facial-landmark-app-v4"
        results = self.runtime_sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        detected_landmarks = get_landmarks_from_response(results)
        self.detectedLandmarks = FacialLandmarks(
            scene=self.imageViewer.graphicsScene,
            view = self.imageViewer.graphicsView,
            landmarks=detected_landmarks,
            sceneWidth=512,
            sceneHeight=512,
            color=Qt.green
        )
        # TODO: plot the landmarks on the image?
        self.draw_detected_landmarks()

    def draw_detected_landmarks(self):
        if self.detectedLandmarks:
            self.detectedLandmarks.draw()

    def replace_landmark(self):
        if self.facialLandmarks:
            self.facialLandmarks.hide()
            self.facialLandmarks.hideIndex()
        self.detectedLandmarks.color = Qt.yellow
        self.facialLandmarks = self.detectedLandmarks
        self.facialLandmarks.draw()

    def set_transparency(self, value):
        alpha = value / 100.0
        # set the transparency of the landmarks according to the slider value
        if self.facialLandmarks:
            self.facialLandmarks.setTransparency(alpha)

    def hide_landmarks(self):
        # Assume self.facialLandmarks.isVisible() checks if landmarks are currently visible
        if self.facialLandmarks.isVisiable:
            self.facialLandmarks.hide()
            self.facialLandmarks.hideIndex()
        else:
            self.facialLandmarks.show()
            self.facialLandmarks.showIndex()

    def discard_image(self, status):
        is_kept = True if status == 0 else False  # status shows the is_discard status, 2 means discard, 0 means keep
        self.csvData_list[self.currentIndex].is_kept = is_kept
        print(f"Clicked discard image switch: {self.csvData_list[self.currentIndex].image_path} with status: {status}, "
              f"is_kept: {self.csvData_list[self.currentIndex].is_kept}")

    def load_control_image_csv(self):
        self.labeling_control_image = True
        csv_file = Path(
            r"C:\Users\Jingwen\Documents\projs\stable-diffusion-webui\avatar_generation\inputs\synthetic_data_info.csv")
        self.csv_path = csv_file
        if not csv_file.exists():
            input_folder = Path(r"C:\Users\Jingwen\Documents\projs\stable-diffusion-webui\avatar_generation\inputs")
            generate_csv(input_folder, csv_file, overwrite=True, type="same_folder")
        with open(csv_file, 'r') as file:
            for line in file:
                if line.startswith('image'):
                    continue
                data = line.split(',')

                self.csvData_list.append(csvRow(data[0], data[1], data[2]))
        self.currentIndex = 0
        print(f"loading 0th image: {self.csvData_list[0].image_path}")
        self.update_curr_img_pose()

    def load_training_image_csv(self):
        self.labeling_control_image = False
        csv_file = Path(
            r"C:\Users\Jingwen\Documents\projs\stable-diffusion-webui\avatar_generation\outputs\synthetic_data_info.csv")
        self.csv_path = csv_file
        if not csv_file.exists():
            input_folder = Path(r"C:\Users\Jingwen\Documents\projs\stable-diffusion-webui\avatar_generation\outputs")
            generate_csv(input_folder, csv_file, overwrite=True)
        with open(csv_file, 'r') as file:
            for line in file:
                if line.startswith('image'):
                    continue
                data = line.split(',')
                self.csvData_list.append(csvRow(data[0], data[1], data[2]))
        self.currentIndex = 0
        self.update_curr_img_pose()

    def save_landmark_template(self):
        print("Saving current Landmark as template")
        if self.facialLandmarks:
            self.landmark_template = self.facialLandmarks
            print(f"Landmark template saved: {self.landmark_template}")

    def load_landmark_template(self):
        print("Loading landmark template")
        if self.facialLandmarks:
            self.facialLandmarks.clear()
        if self.landmark_template:
            temp_pts = np.array(self.landmark_template.getLandmarks())
            # reshape to * x 3
            temp_pts = temp_pts.reshape(-1, 3)

            self.json_dictioinary = {"people": [{}]}
            self.facialLandmarks = FacialLandmarks(
                self.imageViewer.graphicsScene,
                self.imageViewer.graphicsView,
                temp_pts,
                sceneWidth=512,
                sceneHeight=512
            )
            self.facialLandmarks.draw()

    def update_curr_img_pose(self):
        print("Loading image at index:", self.currentIndex)
        print(f"Current image path: {self.csvData_list[self.currentIndex].image_path}")
        if self.currentIndex < 0 or self.currentIndex >= len(self.csvData_list):
            return
        self.imageViewer.load_image(self.csvData_list, self.currentIndex)

        # load the json file and get the landmarks from landmark_path = self.csvData_list[self.currentIndex].landmark
        if Path(self.csvData_list[self.currentIndex].landmark).exists():
            self.load_json(landmark_path=self.csvData_list[self.currentIndex].landmark)
        elif self.landmark_template:
            print("Loading landmark template")
            self.load_landmark_template()
        # else:
        #     print("No landmark json found, no template found, run facial landmark detection")
        #     self.run_facial_landmark_detection()
        #     self.replace_landmark()
        #     self.add_left_pupil()
        #     self.add_right_pupil()


        # get the subfolder and the image name
        currentImagePath = Path(self.csvData_list[self.currentIndex].image_path)

        image_name = currentImagePath.parent.name + "/" + currentImagePath.name
        self.leftControlPanel.imagePathLabel.setText(f"{image_name}")
        self.leftControlPanel.currentIndexLabel.setText(f"Index: {self.currentIndex + 1} / {len(self.csvData_list)}")

        # setup the toggle button values
        self.toggleBboxMode(False)
        self.toggleNose(True)
        self.toggleEyes(True)
        self.toggleMouth(True)
        self.toggleOutline(True)
        self.leftControlPanel.showEyesButton.setChecked(True)
        self.leftControlPanel.showNoseButton.setChecked(True)
        self.leftControlPanel.showMouseButton.setChecked(True)
        self.leftControlPanel.showOutlineButton.setChecked(True)
        self.leftControlPanel.addBboxSwitch.setChecked(False)

        # assign opposite value to the is_discard value of self.csvData_list[self.currentIndex].is_kept
        # strip the string and compare with "True" to get the boolean value
        is_kept = self.csvData_list[self.currentIndex].is_kept
        if type(is_kept) == str:
            is_kept = is_kept.strip().strip('"') == "True"
        # is_kept = self.csvData_list[self.currentIndex].is_kept.strip().strip('"') == "True"

        is_discard = not is_kept
        self.leftControlPanel.discardButton.setChecked(is_discard)

    def prev_image(self):
        if self.facialLandmarks is not None:
            self.save_everything()
        print("Previous image")
        self.currentIndex -= 1
        self.resetScene()
        self.update_curr_img_pose()

    def next_image(self):
        if self.facialLandmarks is not None:
            self.save_everything()
        print("Next image")
        self.currentIndex += 1
        self.resetScene()
        self.update_curr_img_pose()

    def load_json(self, landmark_path):
        if self.currentIndex < 0 or self.currentIndex >= len(self.csvData_list):
            return
        print("landmark_path is %s", landmark_path)
        self.json_dictionary = read_json(landmark_path)
        landmarks, bbox = read_openpose(landmark_path)
        # print(f"Landmarks loaded from file: {landmark_path}:")
        # print(f"Landmarks: {landmarks}")
        scene_width = 512
        scene_height = 512
        if landmarks is not None and len(landmarks) >= 68:
            # Note that here the landmarks are scaled as uv coordinates (0-1)
            self.facialLandmarks = FacialLandmarks(
                self.imageViewer.graphicsScene,
                self.imageViewer.graphicsView,
                landmarks,
                sceneWidth=scene_width,
                sceneHeight=scene_height
            )

            # for i in range(len(self.facialLandmarks.landmarks)):
            #     print(f"keypoint {i}: {i + 24} {self.facialLandmarks.landmarks[i].visibility}")
            self.facialLandmarks.draw()
        else:
            logging.warning("No initial landmark found")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            self.bbox = Bbox(self.imageViewer.graphicsScene)
            self.bbox.createOrUpdate(x1, y1, x2, y2)


    def updateFacialLandmarks(self, scaleFactor):
        currentImageWidth = self.imageViewer.graphicsView.getCurrentImageWidth()
        currentImageHeight = self.imageViewer.getCurrentImageHeight()
        self.facialLandmarks.updateKeypoints(scaleFactor, currentImageWidth, currentImageHeight)

    def save_everything(self):
        if not self.json_dictionary:
            self.json_dictionary = {"people": [{}]}

        if self.facialLandmarks:
            self.save_landmarks_to_file()
        self.save_bbox_to_file()

        self.json_dictionary['canvas_width'] = 512
        self.json_dictionary['canvas_height'] = 512
        self.json_dictionary['canvas_height'] = 512
        # print(self.json_dictionary)
        json_path = self.csvData_list[self.currentIndex].landmark
        print("json path: %s", json_path)
        # Check if the JSON file path is a valid string
        try:
            write_json(self.json_dictionary, json_path)
            print("Landmarks saved to JSON file successfully.")
        except Exception as e:
            logging.error("Error:", e)

    def save_bbox_to_file(self):
        print("Saving bbox")
        self.saveBbox()
        if self.bbox:
            bbox = self.bbox.bbox
            self.json_dictionary["people"][0]['bbox'] = bbox


    def save_landmarks_to_file(self):
        print("Saving landmarks")
        if self.labeling_control_image and not self.facialLandmarks:
            self.replace_landmark()
        landmarks = self.facialLandmarks.getLandmarks()
        currentImagePath = self.csvData_list[self.currentIndex].image_path
        # count number of invisible keypoints
        invisible_count = 0
        for i in range(len(self.facialLandmarks.landmarks)):
            if self.facialLandmarks.landmarks[i].visibility == 1.0:
                invisible_count += 1
        # reset all to visiable if number of invisible keypoints is larger then 30
        if invisible_count > 50:
            for kp in self.facialLandmarks.landmarks:
                kp.visibility = 2

        if "40" in currentImagePath:
            invisible_group = self.facialLandmarks.landmarks[0: 8]
        elif "320" in currentImagePath:
            invisible_group = self.facialLandmarks.landmarks[9: 17]
        else:
            invisible_group = []
        for kp in invisible_group:
            kp.setVisibility(1)



        self.json_dictionary["people"][0]['face_keypoints_2d'] = landmarks


    def setVisibility(self, state):
        if self.facialLandmarks:
            self.facialLandmarks.setVisibility(state)
            self.facialLandmarks.draw()

    def add_left_pupil(self):
        if self.facialLandmarks:
            self.facialLandmarks.addLeftPupil()
            self.facialLandmarks.draw()

    def add_right_pupil(self):
        if self.facialLandmarks:
            self.facialLandmarks.addRightPupil()
            self.facialLandmarks.draw()

    def save_landmark_images(self):
        if self.facialLandmarks:
            input_path = Path(self.csvData_list[self.currentIndex].image_path)
            # add _control_input at the end of the input_path (pathlib.Path object)
            tgt_path = input_path.parent / (input_path.stem + "_controlInput" + input_path.suffix)
            # save the landmarks image to the target path
            self.facialLandmarks.saveImage(tgt_path.as_posix())

    def toggleEyes(self, state):
        """
        Set visibility of eyes, here visibility is not the visibility in annotations, but to show hide them on canvas。
        """
        if self.facialLandmarks:
            self.facialLandmarks.setSceneVisibility(state, "eyes")
            self.facialLandmarks.draw()

    def toggleNose(self, state):
        if self.facialLandmarks:
            self.facialLandmarks.setSceneVisibility(state, "nose")
            self.facialLandmarks.draw()

    def toggleMouth(self, state):
        if self.facialLandmarks:
            self.facialLandmarks.setSceneVisibility(state, "mouth")
            self.facialLandmarks.draw()

    def toggleOutline(self, state):
        if self.facialLandmarks:
            self.facialLandmarks.setSceneVisibility(state, "outline")
            self.facialLandmarks.draw()

    # Add this method to the MainWindow class
    def jump_to_index(self):
        try:
            index = int(self.leftControlPanel.indexInput.text()) - 1  # Convert to 0-based index
            if 0 <= index < len(self.csvData_list):
                print("Jumping to index:", index)
                self.currentIndex = index
                self.update_curr_img_pose()
            else:
                QMessageBox.warning(self, "Error", "Index out of range.")
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number.")

    def selectAll(self):
        if self.facialLandmarks:
            self.facialLandmarks.selectAll()
            self.facialLandmarks.draw()


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



class ImageViewer(QWidget):
    """
    center image and the prev, next button 中央图片显示区域
    """
    requestPreviousImage = pyqtSignal()
    requestNextImage = pyqtSignal()

    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)

        # 使用CustomGraphicsView作为图片显示区域
        self.graphicsView = CustomGraphicsView(self)
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
        self.prevButton.setMinimumSize(30, 40)
        self.nextButton.setMinimumSize(30, 40)

        # 创建水平布局并添加组件
        layout = QHBoxLayout(self)
        layout.addWidget(self.prevButton)
        layout.addWidget(self.graphicsView, 1)
        layout.addWidget(self.nextButton)
        layout.setContentsMargins(0, 0, 0, 0)

        self.prevButton.clicked.connect(self.requestPreviousImage.emit)
        self.nextButton.clicked.connect(self.requestNextImage.emit)
        # 使用QTimer来确保在布局稳定后再调整按钮位置
        QTimer.singleShot(0, self.adjustButtonPositions)

    def load_image(self, csvData_list, currentIndex):
        # 从文件加载图像
        pixmap = QPixmap(csvData_list[currentIndex].image_path)

        scaled_pixmap = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmapItem = QGraphicsPixmapItem(scaled_pixmap)

        self.graphicsScene.addItem(pixmapItem)

        self.graphicsView.fitInView(pixmapItem, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjustButtonPositions()

    def adjustButtonPositions(self):
        initialViewHeight = self.height()

        btnY = initialViewHeight // 2 - self.prevButton.height() // 2
        self.prevButton.move(10, btnY)
        self.nextButton.move(self.width() - self.nextButton.width() - 10, btnY)


class ControlPanel(QWidget):
    def __init__(self, mainWindow, parent=None):
        super(ControlPanel, self).__init__(parent)
        self.maxWidth = 600

        self.setFixedWidth(self.maxWidth + 10)

        self.mainWindow = mainWindow

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # information and index jump session
        self.imagePathLabel = QLabel("")
        self.currentIndexLabel = QLabel("Index: 0 / 0")
        self.indexInput = QLineEdit()
        self.jumpButton = QPushButton("Jump To Index")
        self.imagePathLabel.setWordWrap(True)

        sub_layout1 = QHBoxLayout()
        sub_layout1.addWidget(self.currentIndexLabel)
        sub_layout1.addWidget(self.indexInput)
        sub_layout1.addWidget(self.jumpButton)

        # sliding scale for transparency
        self.transparencySlider = QSlider(Qt.Horizontal)
        self.transparencySlider.setRange(0, 100)
        self.transparencySlider.setValue(100)
        self.transparencySlider.setFixedWidth(self.maxWidth - 10)
        self.hideLandmarkIndexButton = QPushButton("Hide Landmark Index")
        self.runFacialLandmarkDetectionButton = QPushButton("Run Facial Landmark Detection")
        self.replaceButton = QPushButton("Replace Landmark")
        self.loadInitialLandmarksButton = QPushButton("Load Initial Landmarks")
        self.addLeftPupilButton = QPushButton("Add Left Pupil")
        self.addRightPupilButton = QPushButton("Add Right Pupil")

        self.visibleButton = QPushButton('Set Visible (2)', self)
        self.invisibleButton = QPushButton('Set Invisible (1)', self)
        self.nonExistButton = QPushButton('Set Not Existing (0)', self)

        self.selectAllButton = QPushButton("Select All")

        # control landmarks
        self.showEyesButton = self.setupSwitch("Eyes", default=True)
        self.showNoseButton = self.setupSwitch("Nose", default=True)
        self.showMouseButton = self.setupSwitch("Mouth", default=True)
        self.showOutlineButton = self.setupSwitch("Outline", default=True)
        self.saveAsTemplate = QPushButton("Save Landmark and bbox as template")
        self.loadTemplate = QPushButton("Load from template")

        sub_layout2 = QHBoxLayout()
        sub_layout2.addWidget(self.showEyesButton)
        sub_layout2.addWidget(self.showNoseButton)
        sub_layout2.addWidget(self.showMouseButton)
        sub_layout2.addWidget(self.showOutlineButton)

        self.addBboxSwitch = self.setupSwitch("Add BBox")
        self.saveBboxButton = QPushButton("Load BBox to self.bbox")
        sub_layout3 = QHBoxLayout()
        sub_layout3.addWidget(self.addBboxSwitch)
        sub_layout3.addWidget(self.saveBboxButton)

        self.saveFacialLandmarkImage = QPushButton("Save Facial Landmark Image")
        self.discardButton = self.setupSwitch("Discard Image", default=False)
        self.saveButton = QPushButton("Save Bbox and Landmarks in JSON")

        # informations
        info_buttons = [
            self.imagePathLabel,
        ]
        # buttons
        landmark_buttons = [
            self.transparencySlider,
            self.hideLandmarkIndexButton,
            self.runFacialLandmarkDetectionButton,
            self.replaceButton,
            self.addLeftPupilButton,
            self.addRightPupilButton,
            self.nonExistButton,
            self.invisibleButton,
            self.visibleButton,
            self.selectAllButton,
            self.saveAsTemplate,
            self.loadTemplate
        ]
        io_buttons = [self.discardButton, self.saveFacialLandmarkImage, self.saveButton]

        ##################### Layout ############################
        self.setupSection("Information", info_buttons, sub_layout1)
        self.setupSection("Facial Landmarks", landmark_buttons, sub_layout2)
        self.setupSection("Bbox", [], sub_layout3)
        self.setupSection("Export", io_buttons)



        self.layout.setSpacing(mainWindow.hButtonSpace)  # 设置控件之间的间距
        self.layout.setContentsMargins(10, 10, 10, 10)  # 设置布局的边距
        self.layout.addStretch(mainWindow.hStretch)

        self.transparencySlider.valueChanged.connect(self.mainWindow.set_transparency)
        self.jumpButton.clicked.connect(self.mainWindow.jump_to_index)

        self.runFacialLandmarkDetectionButton.clicked.connect(self.mainWindow.run_facial_landmark_detection)
        self.hideLandmarkIndexButton.clicked.connect(self.mainWindow.hide_landmarks)
        self.replaceButton.clicked.connect(self.mainWindow.replace_landmark)

        self.addLeftPupilButton.clicked.connect(self.mainWindow.add_left_pupil)
        self.addRightPupilButton.clicked.connect(self.mainWindow.add_right_pupil)

        self.nonExistButton.clicked.connect(lambda: self.mainWindow.setVisibility(0))
        self.invisibleButton.clicked.connect(lambda: self.mainWindow.setVisibility(1))
        self.visibleButton.clicked.connect(lambda: self.mainWindow.setVisibility(2))

        self.selectAllButton.clicked.connect(lambda: self.mainWindow.selectAll())
        self.saveAsTemplate.clicked.connect(self.mainWindow.save_landmark_template)
        self.loadTemplate.clicked.connect(self.mainWindow.load_landmark_template)


        self.showEyesButton.stateChanged.connect(self.mainWindow.toggleEyes)
        self.showNoseButton.stateChanged.connect(self.mainWindow.toggleNose)
        self.showMouseButton.stateChanged.connect(self.mainWindow.toggleMouth)
        self.showOutlineButton.stateChanged.connect(self.mainWindow.toggleOutline)

        self.addBboxSwitch.stateChanged.connect(self.mainWindow.toggleBboxMode)
        self.saveBboxButton.clicked.connect(self.mainWindow.saveBbox)

        self.saveFacialLandmarkImage.clicked.connect(self.mainWindow.save_landmark_images)
        self.discardButton.stateChanged.connect(self.mainWindow.discard_image)
        self.saveButton.clicked.connect(self.mainWindow.save_everything)

    def setupDivider(self, name):
        label = QLabel(name)
        self.layout.addWidget(label)
        h_divider = QFrame()
        h_divider.setFrameShape(QFrame.HLine)
        h_divider.setFrameShadow(QFrame.Sunken)
        h_divider.setFixedWidth(self.maxWidth + 10)
        self.layout.addWidget(h_divider)

    def setupSection(self, name, widget_list, sub_layout=None):
        self.setupDivider(name)
        for button in widget_list:
            self.layout.addWidget(button)
            button.setFixedWidth(self.maxWidth)
        if sub_layout:
            self.layout.addLayout(sub_layout)
        self.layout.addSpacing(30)

    def setupSwitch(self, name, default=False):
        switch = QCheckBox(name, self)
        switch.setCheckable(True)
        # set default value
        switch.setChecked(default)
        switch.setStyleSheet("""
            QCheckBox::indicator {
                width: 40px;
                height: 20px;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d7;
            }
            QCheckBox::indicator:unchecked {
                background-color: #cccccc;
            }
        """)
        return switch


def setup_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)


if __name__ == '__main__':
    setup_logger()
    app = QApplication(sys.argv)
    viewer = MainWindow()
    viewer.show()
    # viewer.load_csv()
    sys.exit(app.exec_())

from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QGraphicsScene, QPushButton, QHBoxLayout, QGraphicsPixmapItem
from PyQt5.QtCore import Qt

from UI.custom_graphics_view import CustomGraphicsView


class ImageViewer(QWidget):
    """
    center image and the prev, next button 中央图片显示区域
    """
    requestPreviousImage = pyqtSignal()
    requestNextImage = pyqtSignal()

    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.parent = parent

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

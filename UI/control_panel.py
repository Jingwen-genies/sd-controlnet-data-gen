from PyQt5.QtWidgets import QLabel, QFrame, QCheckBox, QPushButton, QHBoxLayout, QSlider, QLineEdit, QVBoxLayout, \
    QWidget
from PyQt5.QtCore import Qt



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

        self.runDetectionForAllButton = QPushButton("Run Detection for All")

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
        io_buttons = [self.discardButton, self.saveFacialLandmarkImage, self.saveButton, self.runDetectionForAllButton]

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
        self.runDetectionForAllButton.clicked.connect(self.mainWindow.run_detection_for_all)

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
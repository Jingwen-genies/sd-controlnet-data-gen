from PyQt5.QtWidgets import QLabel, QFrame, QCheckBox, QPushButton, QHBoxLayout, QSlider, QLineEdit, QVBoxLayout, \
    QWidget, QComboBox, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    selectionChanged = pyqtSignal(str)  # Define a custom signal
    def __init__(self, mainWindow, parent=None):
        super(ControlPanel, self).__init__(parent)
        self.maxWidth = 420

        self.setFixedWidth(self.maxWidth + 10)
        self.mainWindow = mainWindow

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # information and index jump session
        self.imagePathLabel = QLabel("")
        self.currentIndexLabel = QLabel("Index: 0 / 0")
        self.indexInput = QLineEdit()
        self.jumpButton = QPushButton("Jump Index")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.jumpButton.setSizePolicy(sizePolicy)
        self.imagePathLabel.setWordWrap(True)
        self.totalKeptLabel = QLabel("")

        sub_layout1 = QHBoxLayout()
        sub_layout1.addWidget(self.currentIndexLabel)
        sub_layout1.addWidget(self.indexInput)
        sub_layout1.addWidget(self.jumpButton)
        sub_layout1.addWidget(self.totalKeptLabel)

        # sliding scale for transparency
        self.transparencySlider = QSlider(Qt.Horizontal)
        self.transparencySlider.setRange(0, 100)
        self.transparencySlider.setValue(100)
        # self.transparencySlider.setFixedWidth(self.maxWidth - 10)
        self.hideLandmarkIndexButton = QPushButton("Hide Landmark Index")
        self.hideLandmarkIndexButton.setSizePolicy(sizePolicy)
        self.runFacialLandmarkDetectionButton = QPushButton("Run Facial Landmark Detection")
        self.replaceButton = QPushButton("Replace Landmark")
        self.loadInitialLandmarksButton = QPushButton("Load Initial Landmarks")

        self.addLeftPupilButton = QPushButton("Add Left Pupil")
        self.addRightPupilButton = QPushButton("Add Right Pupil")
        sub_layout_pupil = QHBoxLayout()
        sub_layout_pupil.addWidget(self.addLeftPupilButton)
        sub_layout_pupil.addWidget(self.addRightPupilButton)

        self.visibleButton = QPushButton('Set Visible (2)', self)
        self.invisibleButton = QPushButton('Set Invisible (1)', self)
        self.nonExistButton = QPushButton('Set Not Existing (0)', self)
        sub_layout_visibility = QHBoxLayout()
        sub_layout_visibility.addWidget(self.visibleButton)
        sub_layout_visibility.addWidget(self.invisibleButton)
        sub_layout_visibility.addWidget(self.nonExistButton)

        self.selectAllButton = QPushButton("Select All")
        self.unSelectAllButton = QPushButton("Unselect All")
        sub_layout_selection = QHBoxLayout()
        sub_layout_selection.addWidget(self.selectAllButton)
        sub_layout_selection.addWidget(self.unSelectAllButton)

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

        self.runDetectionForAllButton = QPushButton("Run Detection for Batch")
        self.numberOfImagesInput = QLineEdit()
        sub_layout4 = QHBoxLayout()
        sub_layout4.addWidget(self.runDetectionForAllButton)
        sub_layout4.addWidget(self.numberOfImagesInput)

        # model selection
        # Create a label to display the selected option
        sub_layout5 = QVBoxLayout()
        self.endpointLabel = QLabel("Pick an endpoint from the dropdown")
        sub_layout5.addWidget(self.endpointLabel)

        # Create a QComboBox (dropdown menu)
        self.combo_box = QComboBox(self)
        self.combo_box.addItem("facial-landmark-app-v5")
        self.combo_box.addItem("facial-landmark-app-v4")
        self.combo_box.addItem("facial-landmark-app-v2")

        # Connect the selection change event to a handler
        sub_layout5.addWidget(self.combo_box)

        # informations
        info_buttons = [
            self.imagePathLabel,
        ]
        # buttons

        io_buttons = [self.discardButton, self.saveFacialLandmarkImage, self.saveButton]

        sub_layout_landmark_transparency = QHBoxLayout()
        sub_layout_landmark_transparency.addWidget(self.transparencySlider)
        sub_layout_landmark_transparency.addWidget(self.hideLandmarkIndexButton)

        sub_layout_landmark = QHBoxLayout()
        sub_layout_landmark.addWidget(self.runFacialLandmarkDetectionButton)
        sub_layout_landmark.addWidget(self.replaceButton)

        sub_layout_template = QHBoxLayout()
        sub_layout_template.addWidget(self.saveAsTemplate)
        sub_layout_template.addWidget(self.loadTemplate)

        ##################### Layout ############################
        self.setupSection("Information", info_buttons, [sub_layout1])
        self.setupSection(
            "Facial Landmarks",
            [],
            [sub_layout_landmark_transparency, sub_layout_landmark, sub_layout_template, sub_layout_pupil, sub_layout_selection, sub_layout_visibility,  sub_layout2]
        )
        self.setupSection("Bbox", [], [sub_layout3])
        self.setupSection("io", io_buttons, [sub_layout4])
        self.setupSection("Model Selection", [], [sub_layout5])


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
        self.unSelectAllButton.clicked.connect(lambda: self.mainWindow.unSelectAll())
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

        self.combo_box.currentIndexChanged.connect(self.on_combobox_changed)

    def on_combobox_changed(self, index):
        # Emit the custom signal with the selected item
        selected_item = self.combo_box.currentText()
        self.selectionChanged.emit(selected_item)


    def setupDivider(self, name):
        label = QLabel(name)
        self.layout.addWidget(label)
        h_divider = QFrame()
        h_divider.setFrameShape(QFrame.HLine)
        h_divider.setFrameShadow(QFrame.Sunken)
        h_divider.setFixedWidth(self.maxWidth + 10)
        self.layout.addWidget(h_divider)

    def setupSection(self, name, widget_list, sub_layouts=[]):
        self.setupDivider(name)
        for button in widget_list:
            self.layout.addWidget(button)
            button.setFixedWidth(self.maxWidth)
        if sub_layouts:
            for sub_layout in sub_layouts:
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
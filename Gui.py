from PySide6.QtWidgets import (QMainWindow, QStackedWidget, QApplication, QDialog, QLabel,
                               QPushButton, QVBoxLayout, QWidget, QFormLayout, QSpinBox,
                               QDoubleSpinBox)
from PySide6.QtCore import Qt
import cv2
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pi Vision Control Panel")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home = HomePage(self)
        self.recognition = RecognitionPage(self)

        self.stack.addWidget(self.home)
        self.stack.addWidget(self.recognition)

        self.home.start_button.clicked.connect(self.start_recognition)
        self.recognition.stop_button.clicked.connect(self.go_home)

    def start_recognition(self):
        self.stack.setCurrentWidget(self.recognition)
        self.recognition.start_camera()

    def go_home(self):
        self.recognition.stop_camera()
        self.stack.setCurrentWidget(self.home)

class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_button = QPushButton("Start Gesture Recognition")
        self.settings_button = QPushButton("⚙ Settings")
        self.mapping_button = QPushButton("🎮 Gesture Mappings")

        layout.addWidget(self.start_button)
        layout.addWidget(self.settings_button)
        layout.addWidget(self.mapping_button)
        self.setLayout(layout)

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    gesture_ready = Signal(str)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # --- gesture recognition logic ---
            gesture = "Peace"  # Example placeholder
            self.gesture_ready.emit(gesture)
            self.frame_ready.emit(frame)

        cap.release()

class RecognitionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_label = QLabel("Camera Feed")
        self.gesture_label = QLabel("Gesture: ...")
        self.stop_button = QPushButton("⏹ Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.gesture_label)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.thread = CameraThread()
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.gesture_ready.connect(self.update_gesture)

    def start_camera(self):
        self.thread.start()

    def stop_camera(self):
        self.thread.terminate()

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_gesture(self, gesture):
        self.gesture_label.setText(f"Gesture: {gesture}")

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        layout = QFormLayout()

        self.move_distance = QSpinBox()
        self.move_distance.setValue(20)

        self.move_interval = QDoubleSpinBox()
        self.move_interval.setValue(0.03)
        self.move_interval.setSingleStep(0.01)

        self.scroll_amount = QSpinBox()
        self.scroll_amount.setValue(100)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)

        layout.addRow("Mouse move distance (px):", self.move_distance)
        layout.addRow("Move interval (s):", self.move_interval)
        layout.addRow("Scroll amount (px):", self.scroll_amount)
        layout.addWidget(save_button)

        self.setLayout(layout)

class MappingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gesture Mappings")

        layout = QVBoxLayout()
        self.info = QLabel("Click 'Change' then press a key or mouse button.")
        self.change_button = QPushButton("Change 'Peace' mapping")
        layout.addWidget(self.info)
        layout.addWidget(self.change_button)
        self.setLayout(layout)

        self.change_button.clicked.connect(self.start_listening)
        self.listening = False
        self.new_mapping = None

    def start_listening(self):
        self.listening = True
        self.info.setText("Listening for next input...")

    def keyPressEvent(self, event):
        if self.listening:
            key = event.text().upper() or event.key()
            self.new_mapping = f"KEY_{key}"
            self.info.setText(f"Mapped to {self.new_mapping}")
            self.listening = False

    def mousePressEvent(self, event):
        if self.listening:
            self.new_mapping = f"MOUSE_{event.button()}"
            self.info.setText(f"Mapped to {self.new_mapping}")
            self.listening = False

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QApplication, QLabel, QPushButton, QVBoxLayout,
    QWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QHBoxLayout, QDialog,
    QGridLayout, QMessageBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QEvent
from PySide6.QtGui import QImage, QPixmap, QKeySequence
import cv2
import numpy as np
import json
import os
from pathlib import Path
from copy import deepcopy

# ---------- User Settings Management (local to GUI) ----------

DEFAULT_USER_SETTINGS = {
    "MOVE_INTERVAL": 0.03,
    "SCROLL_AMOUNT": 100,
    "MOUSE_SENSITIVITY": 5,
    "MIN_HOLD_FRAMES": 3,
    "MOUSE_HAND": "right",
    "GAME_HAND": "left",
    "MOVE_MARGIN": 30,
    "MAPPINGS": {
        "call": ["esc", "press"],
        "dislike": ["scroll_down", "hold"],
        "fist": ["delete", "press"],
        "four": ["tab", "press"],
        "like": ["scroll_up", "hold"],
        "mute": ["volume_toggle", "press"],
        "ok": ["enter", "press"],
        "one": ["left_click", "hold"],
        "palm": ["space", "press"],
        "peace": ["winleft", "press"],
        "peace_inverted": ["alt", "hold"],
        "rock": ["w", "press"],
        "stop": ["mouse", "move"],
        "stop_inverted": ["game", "hold"],
        "three": ["shift", "hold"],
        "three2": ["left_click", "press"],
        "two_up": ["right_click", "press"],
        "two_up_inverted": ["ctrl", "hold"]
    }
}

def get_config_path():
    """Return the platform-specific path to the PiVision config file."""
    # Windows → AppData\Roaming\PiVision
    # Linux/Mac → ~/.config/PiVision
    if os.name == "nt":
        base_dir = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base_dir = Path.home() / ".config"
    pivision_dir = base_dir / "PiVision"
    pivision_dir.mkdir(parents=True, exist_ok=True)
    return pivision_dir / "config.json"

SETTINGS_FILE = get_config_path()

def load_user_settings():
    """Load settings from local JSON, or create default."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
                # fill in missing keys if new defaults are added later
                for k, v in DEFAULT_USER_SETTINGS.items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception as e:
            print(f"[WARN] Could not read user_settings.json: {e}")
    save_user_settings(DEFAULT_USER_SETTINGS)
    return DEFAULT_USER_SETTINGS.copy()

def save_user_settings(settings):
    """Save settings to local JSON."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"[INFO] User settings saved → {SETTINGS_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save settings: {e}")


# ===============================================================
# -------------------- MAIN WINDOW -------------------------------
# ===============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.user_settings = load_user_settings()
        print("[INFO] Loaded user settings:", self.user_settings)

        self.setWindowTitle("PiVision Control Panel")
        self.resize(1000, 700)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # PAGES
        self.home = HomePage(self)
        self.recognition = RecognitionPage(self)
        self.settings = SettingsPage(self)
        self.mappings = MappingsPage(self)

        # ADD TO STACK
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.recognition)
        self.stack.addWidget(self.settings)
        self.stack.addWidget(self.mappings)

        # CONNECT BUTTONS
        self.home.start_button.clicked.connect(self.start_recognition)
        self.home.settings_button.clicked.connect(self.go_settings)
        self.home.mapping_button.clicked.connect(self.go_mappings)

        self.recognition.stop_button.clicked.connect(self.go_home)
        self.settings.back_button.clicked.connect(self.go_home)
        self.mappings.back_button.clicked.connect(self.go_home)

        # Apply modern dark style
        self.apply_stylesheet()

    # ------------- Navigation --------------
    def start_recognition(self):
        self.stack.setCurrentWidget(self.recognition)
        self.recognition.start_camera()

    def go_home(self):
        self.recognition.stop_camera()
        self.stack.setCurrentWidget(self.home)

    def go_settings(self):
        self.settings.load_settings_from_json()
        self.stack.setCurrentWidget(self.settings)

    def go_mappings(self):
        self.stack.setCurrentWidget(self.mappings)

    # ------------- Stylesheet --------------
    def apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #dddddd;
                font-family: Segoe UI, Roboto, Arial;
                font-size: 15px;
            }

            QLabel {
                color: #f0f0f0;
            }

            QPushButton {
                background-color: #2d2d30;
                color: #ffffff;
                padding: 10px 20px;
                border-radius: 10px;
                border: 1px solid #3e3e42;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a3a3d;
            }
            QPushButton:pressed {
                background-color: #0078d7;
            }

            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2b2b2b;
                border: 1px solid #3e3e42;
                padding: 5px;
                border-radius: 6px;
                color: #ffffff;
            }

            QScrollArea {
                background: transparent;
                border: none;
            }

            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 10px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }

            QHeaderLabel {
                font-size: 22px;
                font-weight: bold;
                color: #ffffff;
            }
        """)
    #------------ Functionality of updating JSON ---------------
    def update_gesture_mapping(self, gesture_name, action_key, action_type):
        """Update one gesture mapping and save to local JSON."""
        active_preset = self.user_settings.get("active_preset", "default")

        # Make sure the preset exists
        if "presets" not in self.user_settings:
            self.user_settings["presets"] = {}
        if active_preset not in self.user_settings["presets"]:
            self.user_settings["presets"][active_preset] = {}

        # Update mapping
        self.user_settings["presets"][active_preset][gesture_name] = [action_key, action_type]

        # Save immediately
        save_user_settings(self.user_settings)
        print(f"[INFO] Updated '{gesture_name}' → {action_key}, {action_type}")


# ===============================================================
# -------------------- HOME PAGE --------------------------------
# ===============================================================

class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("PiVision Control Panel")
        title.setStyleSheet("font-size: 28px; font-weight: bold; margin-bottom: 40px;")

        self.start_button = QPushButton("▶ Start Gesture Recognition")
        self.settings_button = QPushButton("⚙ Settings")
        self.mapping_button = QPushButton("🎮 Gesture Mappings")

        for btn in [self.start_button, self.settings_button, self.mapping_button]:
            btn.setFixedWidth(300)
            btn.setFixedHeight(50)

        layout.addWidget(title)
        layout.addWidget(self.start_button)
        layout.addWidget(self.settings_button)
        layout.addWidget(self.mapping_button)
        layout.setSpacing(20)
        self.setLayout(layout)

# ===============================================================
# -------------------- CAMERA THREAD -----------------------------
# ===============================================================

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    gesture_ready = Signal(str)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gesture = "Peace"
            self.gesture_ready.emit(gesture)
            self.frame_ready.emit(frame)
        cap.release()

# ===============================================================
# -------------------- RECOGNITION PAGE --------------------------
# ===============================================================

class RecognitionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        header = QLabel("🖐 Gesture Recognition")
        header.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 20px;")

        self.video_label = QLabel("Camera Feed")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 10px;")

        self.gesture_label = QLabel("Gesture: ...")
        self.gesture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stop_button = QPushButton("⏹ Stop Recognition")

        layout.addWidget(header, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.gesture_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stop_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

        self.setLayout(layout)

        self.thread = CameraThread()
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.gesture_ready.connect(self.update_gesture)

    def start_camera(self):
        self.thread.start()

    def stop_camera(self):
        if self.thread.isRunning():
            self.thread.terminate()

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_gesture(self, gesture):
        self.gesture_label.setText(f"Gesture: {gesture}")

# ===============================================================
# -------------------- SETTINGS PAGE -----------------------------
# ===============================================================

class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.user_settings = self.parent_window.user_settings

        layout = QVBoxLayout()
        header = QLabel("⚙ Settings")
        header.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 20px;")

        # --- Form Layout ---
        form = QFormLayout()

        # Mouse sensitivity
        self.mouse_sensitivity = QDoubleSpinBox()
        self.mouse_sensitivity.setRange(0.1, 10.0)
        self.mouse_sensitivity.setSingleStep(0.1)
        form.addRow("Mouse Sensitivity (0.1 - 10.0):", self.mouse_sensitivity)

        # Scroll speed
        self.scroll_amount = QSpinBox()
        self.scroll_amount.setRange(10, 1000)
        form.addRow("Scroll Speed (px, 10 - 1000):", self.scroll_amount)

        # Min hold frames
        self.min_hold_frames = QSpinBox()
        self.min_hold_frames.setRange(1, 120)
        form.addRow("Minimum Hold Frames (1 - 120):", self.min_hold_frames)

        # Mouse hand
        self.mouse_hand = QComboBox()
        self.mouse_hand.addItems(["right", "left"])
        form.addRow("Mouse Hand:", self.mouse_hand)

        # Game hand
        self.game_hand = QComboBox()
        self.game_hand.addItems(["right", "left"])
        form.addRow("Game Hand:", self.game_hand)

        # Move Interval
        self.move_interval = QDoubleSpinBox()
        self.move_interval.setRange(0.001, 1)
        self.move_interval.setSingleStep(0.01)
        form.addRow("Move Interval (0.001 - 1.0):", self.move_interval)

        # Move Margin
        self.move_margin = QDoubleSpinBox()
        self.move_margin.setRange(1, 100)
        self.move_margin.setSingleStep(5)
        form.addRow("Move Margin (1 - 100.0):", self.move_margin)

        # --- Buttons ---
        self.save_button = QPushButton("💾 Save Settings")
        self.revert_button = QPushButton("↩ Revert to Default")
        self.back_button = QPushButton("← Back")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.revert_button)
        button_layout.addWidget(self.back_button)

        layout.addWidget(header, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(form)
        layout.addSpacing(20)
        layout.addLayout(button_layout)
        layout.addStretch()
        self.setLayout(layout)

        # Connect signals
        self.save_button.clicked.connect(self.save_settings)
        self.revert_button.clicked.connect(self.revert_to_default)

        # Load initial data
        self.load_settings_from_json()

    # ------------------------------
    # Load current JSON values
    # ------------------------------
    def load_settings_from_json(self):
        """Read user settings and populate input widgets."""
        settings = self.parent_window.user_settings
        self.mouse_sensitivity.setValue(settings.get("MOUSE_SENSITIVITY", 5))
        self.scroll_amount.setValue(settings.get("SCROLL_AMOUNT", 100))
        self.min_hold_frames.setValue(settings.get("MIN_HOLD_FRAMES", 3))
        self.mouse_hand.setCurrentText(settings.get("MOUSE_HAND", "right"))
        self.game_hand.setCurrentText(settings.get("GAME_HAND", "left"))
        self.move_interval.setValue(settings.get("MOVE_INTERVAL", 0.03))
        self.move_margin.setValue(settings.get("MOVE_MARGIN", 30))

    # ------------------------------
    # Save updated values to JSON
    # ------------------------------
    def save_settings(self):
        """Update user_settings.json with the new values."""
        settings = self.parent_window.user_settings

        settings["MOUSE_SENSITIVITY"] = self.mouse_sensitivity.value()
        settings["SCROLL_AMOUNT"] = self.scroll_amount.value()
        settings["MIN_HOLD_FRAMES"] = self.min_hold_frames.value()
        settings["MOUSE_HAND"] = self.mouse_hand.currentText().lower()
        settings["GAME_HAND"] = self.game_hand.currentText().lower()
        settings["MOVE_INTERVAL"] = self.move_interval.value()
        settings["MOVE_MARGIN"] = self.move_margin.value()

        save_user_settings(settings)
        self.parent_window.user_settings = load_user_settings()

        QMessageBox.information(self, "Settings Saved", "Your settings were successfully saved!")

    # ------------------------------
    # Revert to default values
    # ------------------------------
    def revert_to_default(self):
        """Reset settings to their default values."""
        default_settings = {
            "MOVE_INTERVAL": 0.03,
            "SCROLL_AMOUNT": 100,
            "MOUSE_SENSITIVITY": 5,
            "MIN_HOLD_FRAMES": 3,
            "MOUSE_HAND": "right",
            "GAME_HAND": "left",
            "MOVE_MARGIN": 30,
            "gesture_mappings": {  # ensure this field stays intact
                "Swipe Up": {"action": "scroll_up", "mode": "short"},
                "Swipe Down": {"action": "scroll_down", "mode": "short"},
                "Swipe Left": {"action": "go_back", "mode": "short"},
                "Swipe Right": {"action": "go_forward", "mode": "short"}
            }
        }

        reply = QMessageBox.question(
            self, "Confirm Reset",
            "Are you sure you want to reset all settings to default?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        save_user_settings(default_settings)
        self.parent_window.user_settings = load_user_settings()
        self.load_settings_from_json()

        QMessageBox.information(self, "Reverted", "Settings have been reset to default values.")


# ===============================================================
# -------------------- MAPPINGS PAGE -----------------------------
# ===============================================================

class MappingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # keep parent ref for saving
        self.parent_window = parent

        layout = QVBoxLayout()
        header = QLabel("🎮 Gesture Mappings")
        header.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(header, alignment=Qt.AlignmentFlag.AlignCenter)

        # Preset selector (kept for UI; active_preset is used)
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_selector = QComboBox()
        # fill presets from user_settings if available
        presets = list(self.parent_window.user_settings.get("presets", {}).keys())
        if not presets:
            presets = ["default"]
        self.preset_selector.addItems(presets)
        self.preset_selector.setCurrentText(self.parent_window.user_settings.get("active_preset", presets[0]))
        self.preset_selector.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_selector)
        layout.addLayout(preset_layout)

        # Scroll area for mappings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.grid = QGridLayout()
        container.setLayout(self.grid)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # bottom buttons
        btn_layout = QHBoxLayout()
        self.revert_button = QPushButton("↩ Revert to Default")
        self.back_button = QPushButton("← Back")
        btn_layout.addWidget(self.revert_button)
        btn_layout.addWidget(self.back_button)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # storage for widget refs
        self.bind_buttons = {}     # gesture -> QPushButton (shows current binding)
        self.duration_boxes = {}   # gesture -> QComboBox (press/hold)
        self.listening_gesture = None  # currently listening gesture name or None

        # populate UI
        self.populate_from_settings()

        # wire buttons
        self.revert_button.clicked.connect(self.revert_mappings)
        # back_button already connected by MainWindow in your setup

        # install event filter on application to capture keys & mouse while listening
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

    # ---------------- UI population ----------------
    def populate_from_settings(self):
        # clear grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # header row
        self.grid.addWidget(QLabel("Gesture"), 0, 0)
        self.grid.addWidget(QLabel("Binding"), 0, 1)
        self.grid.addWidget(QLabel("Mode"), 0, 2)
        self.grid.addWidget(QLabel("Info"), 0, 3)

        # get current preset data
        preset_name = self.preset_selector.currentText()
        presets = self.parent_window.user_settings.get("presets", {})
        preset_data = presets.get(preset_name, {})

        gestures = sorted(preset_data.keys())
        for i, g in enumerate(gestures, start=1):
            lbl = QLabel(g)

            # binding button (click to listen)
            btn = QPushButton(self.display_text_for_action(preset_data.get(g, ["custom_key", "press"])[0]))
            btn.setToolTip("Click to change binding; then press a key or mouse button.")
            btn.clicked.connect(lambda _, gesture=g: self.start_listening_for(gesture))
            btn.setFixedWidth(220)

            # duration combo
            mode_combo = QComboBox()
            mode_combo.addItems(["press", "hold"])
            current_mode = preset_data.get(g, ["", "press"])[1]
            mode_combo.setCurrentText(current_mode if current_mode in ("press", "hold") else "press")
            mode_combo.currentTextChanged.connect(lambda _, gesture=g: self.on_mode_changed(gesture))

            # info button
            info = QPushButton("🛈")
            info.setFixedWidth(60)
            info.clicked.connect(lambda _, gesture=g: self.show_info(gesture))

            self.grid.addWidget(lbl, i, 0)
            self.grid.addWidget(btn, i, 1)
            self.grid.addWidget(mode_combo, i, 2)
            self.grid.addWidget(info, i, 3)

            # store refs
            self.bind_buttons[g] = btn
            self.duration_boxes[g] = mode_combo

    # ---------------- helpers ----------------
    def on_preset_changed(self, preset_name):
        # update active preset in memory and reload UI
        self.parent_window.user_settings["active_preset"] = preset_name
        save_user_settings(self.parent_window.user_settings)
        self.populate_from_settings()

    def display_text_for_action(self, action_key: str) -> str:
        """Make a human-friendly label for a raw action string."""
        if not action_key:
            return "Unassigned"
        ak = action_key.lower()
        mapping = {
            "left_click": "Left Click",
            "right_click": "Right Click",
            "middle_click": "Middle Click",
            "scroll_up": "Scroll Up",
            "scroll_down": "Scroll Down"
        }
        if ak in mapping:
            return mapping[ak]
        # typical keyboard string (e.g. "ctrl+a" or "space")
        return action_key

    def start_listening_for(self, gesture_name):
        """Begin listening for next key or mouse button for given gesture."""
        # if already listening something, ignore or replace
        if self.listening_gesture is not None:
            QMessageBox.information(self, "Listening", f"Already listening for '{self.listening_gesture}'. Press a key or cancel.")
            return
        self.listening_gesture = gesture_name
        # update button visual
        btn = self.bind_buttons.get(gesture_name)
        if btn:
            btn.setText("Listening... (press key or click mouse)")
            btn.setStyleSheet("background-color: #444; font-weight: bold;")
        # give user focus hint
        self.grabKeyboard()
        self.grabMouse()

    def stop_listening(self):
        """Stop listening mode."""
        if self.listening_gesture:
            # restore button text from current settings
            preset = self.parent_window.user_settings.get("active_preset", "default")
            action = self.parent_window.user_settings.get("presets", {}).get(preset, {}).get(self.listening_gesture, [""])[0]
            btn = self.bind_buttons.get(self.listening_gesture)
            if btn:
                btn.setText(self.display_text_for_action(action))
                btn.setStyleSheet("")
        self.listening_gesture = None
        try:
            self.releaseKeyboard()
            self.releaseMouse()
        except Exception:
            pass

    def on_mode_changed(self, gesture_name):
        """When press/hold changed — save mapping immediately (keep action the same)."""
        preset = self.parent_window.user_settings.get("active_preset", "default")
        action = self.parent_window.user_settings.get("presets", {}).get(preset, {}).get(gesture_name, ["unassigned", "press"])[0]
        mode = self.duration_boxes[gesture_name].currentText().lower()
        # persist
        self.parent_window.update_gesture_mapping(gesture_name, action, mode)

    def save_mappings(self):
        # ensure saved (MainWindow.update_gesture_mapping already persisted per change)
        save_user_settings(self.parent_window.user_settings)
        QMessageBox.information(self, "Saved", "Mappings saved to user_settings.json.")

    def revert_mappings(self):
        """Restore the default preset mappings from DEFAULT_USER_SETTINGS."""
        confirm = QMessageBox.question(
            self, "Revert Mappings",
            "Are you sure you want to revert all mappings to default?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        # deep copy default preset to avoid shared references
        default_presets = deepcopy(DEFAULT_USER_SETTINGS.get("presets", {}))

        # overwrite user_settings
        self.parent_window.user_settings["presets"] = default_presets
        self.parent_window.user_settings["active_preset"] = "default"

        # update preset selector in UI
        self.preset_selector.clear()
        self.preset_selector.addItems(list(default_presets.keys()))
        self.preset_selector.setCurrentText("default")

        save_user_settings(self.parent_window.user_settings)
        QMessageBox.information(self, "Reverted", "Mappings reverted to default.")
        self.populate_from_settings()

    def show_info(self, gesture):
        QMessageBox.information(self, "Gesture Preview", f"Preview for '{gesture}' gesture (image/video to be added).")

    # ---------------- event filter to capture input ----------------
    def eventFilter(self, obj, event):
        # only act when listening
        if self.listening_gesture is None:
            return super().eventFilter(obj, event)

        # capture keyboard events
        if event.type() == QEvent.KeyPress:
            try:
                action_key = self.key_event_to_action(event)
                mode = self.duration_boxes[self.listening_gesture].currentText().lower()
                self.parent_window.update_gesture_mapping(self.listening_gesture, action_key, mode)
                btn = self.bind_buttons.get(self.listening_gesture)
                if btn:
                    btn.setText(self.display_text_for_action(action_key))
                self.stop_listening()
                return True  # consume event
            except Exception as e:
                print("Error capturing key event:", e)
                self.stop_listening()
                return True

        # capture mouse button events
        if event.type() == QEvent.MouseButtonPress:
            try:
                action_key = self.mouse_event_to_action(event)
                mode = self.duration_boxes[self.listening_gesture].currentText().lower()
                self.parent_window.update_gesture_mapping(self.listening_gesture, action_key, mode)
                btn = self.bind_buttons.get(self.listening_gesture)
                if btn:
                    btn.setText(self.display_text_for_action(action_key))
                self.stop_listening()
                return True
            except Exception as e:
                print("Error capturing mouse event:", e)
                self.stop_listening()
                return True

        return super().eventFilter(obj, event)

    def key_event_to_action(self, event):
        """Convert a QKeyEvent into a canonical action string (e.g. 'ctrl+a', 'space', 'esc')."""
        # modifiers
        parts = []
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            parts.append("ctrl")
        if mods & Qt.AltModifier:
            parts.append("alt")
        if mods & Qt.ShiftModifier:
            parts.append("shift")
        if mods & Qt.MetaModifier:
            parts.append("meta")

        # get a readable key string
        text = event.text()
        if text and text.strip():
            keyname = text.lower()
        else:
            # fallback to common special keys
            key = event.key()
            mapping = {
                Qt.Key_Escape: "esc",
                Qt.Key_Space: "space",
                Qt.Key_Return: "enter",
                Qt.Key_Enter: "enter",
                Qt.Key_Backspace: "backspace",
                Qt.Key_Tab: "tab",
                Qt.Key_Delete: "delete",
                Qt.Key_Left: "left",
                Qt.Key_Right: "right",
                Qt.Key_Up: "up",
                Qt.Key_Down: "down",
            }
            keyname = mapping.get(key, QKeySequence(key).toString().lower() or f"key_{int(key)}")

        if parts:
            return "+".join(parts + [keyname])
        return keyname

    def mouse_event_to_action(self, event):
        """Convert QMouseEvent into canonical action string."""
        btn = event.button()
        if btn == Qt.LeftButton:
            return "left_click"
        if btn == Qt.RightButton:
            return "right_click"
        if btn == Qt.MiddleButton:
            return "middle_click"
        return "mouse_button"

    # ensure cleanup: uninstall filter if widget destroyed
    def closeEvent(self, ev):
        app = QApplication.instance()
        if app:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        super().closeEvent(ev)


# ===============================================================
# -------------------- MAIN ENTRY --------------------------------
# ===============================================================

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

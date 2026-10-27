import math
import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

SCALE = 2          # fog is computed at 1/SCALE resolution (higher = faster, blockier)
EDGE_SOFTNESS = 5  # blur applied to the fog edge at full resolution


def resource_path(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def is_pointing(hand):
    wrist = hand[0]

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def extended(tip, pip):
        return dist(wrist, hand[tip]) > dist(wrist, hand[pip]) * 1.1

    index_out = extended(8, 6)
    others_curled = not any(extended(t, p) for t, p in [(12, 10), (16, 14), (20, 18)])
    return index_out and others_curled


class Worker(QThread):
    frame_ready = Signal(QImage)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        # Settings the UI changes while running
        self.brush = 40
        self.fog_strength = 0.6   # 0 = no fog, 1 = solid white
        self.regrow = 0.003
        self.smoothing = 0.5
        self.paused = False
        self.clear_requested = False
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=resource_path("hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.error.emit("Could not open the camera. Check camera permission "
                            "in System Settings > Privacy & Security > Camera.")
            return

        fog = None
        prev_tips = {}
        start = time.monotonic()

        while self._running:
            if self.paused:
                self.msleep(50)
                continue

            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror
            h, w = frame.shape[:2]
            sw, sh = w // SCALE, h // SCALE

            if fog is None:
                fog = np.ones((sh, sw), dtype=np.float32)
            if self.clear_requested:
                fog[:] = 1.0
                self.clear_requested = False

            # Detect hands (video mode needs increasing timestamps in ms)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int((time.monotonic() - start) * 1000)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            # Wipe fog along the smoothed fingertip path
            brush = max(1, int(self.brush) // SCALE)
            seen = set()
            for i, hand in enumerate(result.hand_landmarks):
                if not is_pointing(hand):
                    continue
                seen.add(i)
                x, y = hand[8].x * w, hand[8].y * h
                if i in prev_tips:
                    px, py = prev_tips[i]
                    x = px + self.smoothing * (x - px)
                    y = py + self.smoothing * (y - py)
                    end = (int(x / SCALE), int(y / SCALE))
                    cv2.line(fog, (int(px / SCALE), int(py / SCALE)), end, 0.0, brush)
                    cv2.circle(fog, end, max(1, brush // 2), 0.0, -1)  # round end
                else:
                    cv2.circle(fog, (int(x / SCALE), int(y / SCALE)),
                               max(1, brush // 2), 0.0, -1)
                prev_tips[i] = (x, y)
            for i in list(prev_tips):
                if i not in seen:
                    del prev_tips[i]

            # Fog slowly creeps back
            if self.regrow > 0:
                fog += self.regrow
                np.minimum(fog, 1.0, out=fog)

            # Foggy frame: blur + wash toward white, at low resolution
            s = self.fog_strength
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (13, 13), 0)
            small = cv2.convertScaleAbs(small, alpha=1 - s, beta=255 * s)
            foggy = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

            # Upscale the mask smoothly, then soften the edge at full resolution
            alpha = cv2.resize(fog, (w, h), interpolation=cv2.INTER_CUBIC)
            alpha = cv2.GaussianBlur(alpha, (0, 0), EDGE_SOFTNESS)
            alpha = np.clip(alpha, 0.0, 1.0)
            out = cv2.blendLinear(frame, foggy, 1.0 - alpha, alpha)

            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img = QImage(out_rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(img)

        cap.release()
        detector.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Foggy Mirror")
        self.last_image = None

        self.video = QLabel("Starting camera...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(320, 240)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet("background: #111; color: #aaa;")

        self.worker = Worker()
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.error.connect(self.show_error)

        panel = QVBoxLayout()
        panel.addWidget(self.make_slider(
            "Brush size", 10, 100, 40, lambda v: setattr(self.worker, "brush", v)))
        panel.addWidget(self.make_slider(
            "Fog thickness", 0, 100, 60,
            lambda v: setattr(self.worker, "fog_strength", v / 100)))
        panel.addWidget(self.make_slider(
            "Fog regrow speed", 0, 100, 15,
            lambda v: setattr(self.worker, "regrow", v * 0.0002)))
        panel.addWidget(self.make_slider(
            "Smoothing", 10, 100, 50,
            lambda v: setattr(self.worker, "smoothing", v / 100)))

        refill = QPushButton("Refill fog")
        refill.clicked.connect(lambda: setattr(self.worker, "clear_requested", True))
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        save = QPushButton("Save screenshot")
        save.clicked.connect(self.save_screenshot)
        for b in (refill, self.pause_btn, save):
            panel.addWidget(b)
        panel.addStretch()

        side = QWidget()
        side.setLayout(panel)
        side.setFixedWidth(220)

        layout = QHBoxLayout()
        layout.addWidget(self.video, 1)
        layout.addWidget(side)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(940, 520)

        self.worker.start()

    def make_slider(self, title, lo, hi, value, on_change):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"{title}: {value}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)

        def changed(v):
            label.setText(f"{title}: {v}")
            on_change(v)

        slider.valueChanged.connect(changed)
        lay.addWidget(label)
        lay.addWidget(slider)
        return box

    def show_frame(self, img):
        self.last_image = img
        pix = QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pix)

    def show_error(self, msg):
        QMessageBox.critical(self, "Foggy Mirror", msg)

    def toggle_pause(self):
        self.worker.paused = not self.worker.paused
        self.pause_btn.setText("Resume" if self.worker.paused else "Pause")

    def save_screenshot(self):
        if self.last_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save screenshot", "foggy_mirror.png", "PNG image (*.png)")
        if path:
            self.last_image.save(path)

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
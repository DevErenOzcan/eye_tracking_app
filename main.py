import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QLabel
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush, QImage, QPixmap

# OpenCV ve PyQt5 arasındaki plugin çakışmasını önler
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    os.path.dirname(sys.modules["PyQt5"].__file__), "Qt", "plugins"
)

# ---------------------------------------------------------
# 1. GÖZ TAKİBİ VE KAMERA İŞLEME THREAD'İ
# ---------------------------------------------------------
class VideoThread(QThread):
    gaze_signal = pyqtSignal(float, float, float, float, QImage)

    def run(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        cap = cv2.VideoCapture(0)

        self.calibrated_center = None
        self.eye_width_ref = 10.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            dx, dy, ix, iy = 0, 0, 0, 0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in landmarks])

                cv2.polylines(frame, [mesh_points[33:133]], True, (0, 255, 0), 1, cv2.LINE_AA)

                iris_left = mesh_points[468]
                iris_right = mesh_points[473]

                cv2.circle(frame, iris_left, 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, iris_right, 4, (0, 0, 255), -1, cv2.LINE_AA)

                left_eye_inner = mesh_points[33]
                left_eye_outer = mesh_points[133]
                current_eye_width = np.linalg.norm(left_eye_inner - left_eye_outer)

                ix, iy = iris_left[0], iris_left[1]

                if self.calibrated_center is not None:
                    scale_factor = current_eye_width / self.eye_width_ref
                    raw_dx = ix - self.calibrated_center[0]
                    raw_dy = iy - self.calibrated_center[1]

                    dx = raw_dx / scale_factor
                    dy = raw_dy / scale_factor

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            self.gaze_signal.emit(dx, dy, float(ix), float(iy), qt_image)

        cap.release()

    def set_calibration(self, center_x, center_y, eye_w):
        self.calibrated_center = (center_x, center_y)
        self.eye_width_ref = eye_w


# ---------------------------------------------------------
# 2. RADYAL KLAVYE WIDGET'I
# ---------------------------------------------------------
class RadialKeyboard(QWidget):
    key_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.dx = 0
        self.dy = 0
        self.angle = 0
        self.magnitude = 0

        self.current_mode = "lowercase"
        self.in_mode_select = False
        self.is_calibrating = True
        self.calibration_progress = 0

        self.deadzone_radius = 3.5
        self.dwell_max = 20

        self.dwell_timer = QTimer()
        self.dwell_timer.timeout.connect(self.trigger_selection)
        self.hovered_key = None
        self.dwell_progress = 0

        base_keys = ["SPACE", "BACK", "ENTER", "MODE"]
        self.layouts = {
            "lowercase": list("abcdefghijklmnopqrstuvwxyz") + base_keys,
            "uppercase": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + base_keys,
            "numbers": list("0123456789+-*/=.,") + base_keys,
            "symbols": list("!@#$%^&()_[]{}:;<>?|\\~") + base_keys
        }

        # Animasyon Timer'ı
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_dwell)
        self.anim_timer.start(50)

    def update_gaze_data(self, dx, dy):
        self.dx = dx
        self.dy = dy
        self.angle = math.degrees(math.atan2(dy, dx))
        self.magnitude = math.sqrt(dx ** 2 + dy ** 2)
        self.update()

    def update_dwell(self):
        if self.is_calibrating:
            self.hovered_key = None
            self.dwell_progress = 0
            self.update()
            return

        if self.magnitude < self.deadzone_radius:
            self.hovered_key = None
            self.dwell_progress = 0
            self.update()
            return

        current_key = self.get_key_at_angle(self.angle)

        if current_key == self.hovered_key:
            self.dwell_progress += 1
        else:
            self.hovered_key = current_key
            self.dwell_progress = 0

        if self.dwell_progress >= self.dwell_max:
            self.trigger_selection()
            self.dwell_progress = 0
            self.hovered_key = None

        self.update()

    def get_key_at_angle(self, angle):
        deg = angle
        if deg < 0: deg += 360

        if self.in_mode_select:
            if 315 <= deg or deg < 45:
                return "numbers"  # Sağ
            elif 45 <= deg < 135:
                return "symbols"  # Alt
            elif 135 <= deg < 225:
                return "lowercase"  # Sol
            return "uppercase"  # Üst
        else:
            keys = self.layouts[self.current_mode]
            step = 360 / len(keys)
            index = int(deg / step)
            if index >= len(keys): index = len(keys) - 1
            return keys[index]

    def trigger_selection(self):
        if not self.hovered_key: return

        if self.hovered_key == "MODE":
            self.in_mode_select = True
            self.dwell_progress = 0
            self.key_selected.emit("[MOD SEÇİMİ AÇIK]")
        elif self.in_mode_select:
            self.current_mode = self.hovered_key
            self.in_mode_select = False
            self.key_selected.emit(f"[MOD DEĞİŞTİ: {self.current_mode.upper()}]")
        else:
            self.key_selected.emit(self.hovered_key)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        center = QPointF(w / 2, h / 2)
        radius = min(w, h) / 2 - 20

        if self.is_calibrating: painter.setOpacity(0.3)

        if self.in_mode_select:
            self.draw_mode_selector(painter, center, radius)
        else:
            self.draw_keys(painter, center, radius)

        painter.setOpacity(1.0)

        if self.is_calibrating:
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.setBrush(Qt.NoBrush)
            target_radius = 60
            painter.drawEllipse(center, target_radius, target_radius)

            painter.setBrush(QColor(0, 255, 0))
            painter.setPen(Qt.NoPen)
            fill_radius = target_radius * (self.calibration_progress / 100.0)
            painter.drawEllipse(center, fill_radius, fill_radius)

            painter.setPen(Qt.white)
            painter.setFont(QFont("Arial", 14, QFont.Bold))
            painter.drawText(QRectF(center.x() - 100, center.y() - 100, 200, 200), Qt.AlignCenter, "MERKEZE\nODAKLAN")
        else:
            vis_deadzone = 50
            painter.setBrush(QColor(30, 30, 30))
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            painter.drawEllipse(center, vis_deadzone, vis_deadzone)

            visual_scale = 18.0  # Hassasiyet çarpanı
            cursor_x = center.x() + (self.dx * visual_scale)
            cursor_y = center.y() + (self.dy * visual_scale)

            dist = math.sqrt((cursor_x - center.x()) ** 2 + (cursor_y - center.y()) ** 2)
            if dist > radius:
                angle_rad = math.atan2(cursor_y - center.y(), cursor_x - center.x())
                cursor_x = center.x() + radius * math.cos(angle_rad)
                cursor_y = center.y() + radius * math.sin(angle_rad)

            if self.magnitude < self.deadzone_radius:
                painter.setBrush(QColor(100, 100, 100))  # Gri
            else:
                painter.setBrush(QColor(0, 255, 0))  # Yeşil

            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cursor_x, cursor_y), 12, 12)

    def draw_keys(self, painter, center, radius):
        keys = self.layouts[self.current_mode]
        step = 360 / len(keys)
        for i, key in enumerate(keys):
            angle = i * step
            is_hover = (key == self.hovered_key)
            self.draw_pie_slice(painter, center, radius, -angle, -step, key, is_hover)

    def draw_mode_selector(self, painter, center, radius):
        self.draw_pie_slice(painter, center, radius, 90, 90, "UPPER", self.hovered_key == "uppercase")
        self.draw_pie_slice(painter, center, radius, 180, 90, "LOWER", self.hovered_key == "lowercase")
        self.draw_pie_slice(painter, center, radius, 270, 90, "SYMBL", self.hovered_key == "symbols")
        self.draw_pie_slice(painter, center, radius, 0, 90, "NUMBR", self.hovered_key == "numbers")

    def draw_pie_slice(self, painter, center, radius, start_angle, span_angle, text, selected=False):
        if selected:
            ratio = self.dwell_progress / self.dwell_max
            color = QColor(0, 255, 0)
            color.setAlpha(100 + int(155 * ratio))
            painter.setBrush(color)
        else:
            painter.setBrush(QColor(50, 50, 50))

        painter.setPen(QPen(Qt.white, 2))
        rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        painter.drawPie(rect, int(start_angle * 16), int(span_angle * 16))

        math_angle = math.radians(start_angle + span_angle / 2)
        text_r = radius * 0.75
        tx = center.x() + text_r * math.cos(math_angle)
        ty = center.y() - text_r * math.sin(math_angle)

        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(QRectF(tx - 30, ty - 15, 60, 30), Qt.AlignCenter, text)


# --- 3. ANA UYGULAMA (PENCERE) ---
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Göz Kontrollü Terminal")
        self.setStyleSheet("background-color: #111; color: #0f0;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.terminal_display = QTextEdit()
        self.terminal_display.setReadOnly(True)
        self.terminal_display.setStyleSheet(
            "background-color: black; color: #00ff00; font-family: Monospace; font-size: 18px; border: 1px solid #333;")
        self.terminal_display.setText("kullanici@goz-pc:~$ ")
        main_layout.addWidget(self.terminal_display, 1)

        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(320, 240)
        self.camera_label.setStyleSheet("border: 2px solid #0f0; background-color: #000;")

        self.camera_label.move(800, 20)
        self.camera_label.raise_()

        self.keyboard = RadialKeyboard()
        self.keyboard.key_selected.connect(self.handle_key_input)
        main_layout.addWidget(self.keyboard, 3)

        self.current_cmd = ""

        self.calibration_buffer = []
        self.calib_timer = QTimer()
        self.calib_timer.timeout.connect(self.collect_calibration_data)

        self.video_thread = VideoThread()
        self.video_thread.gaze_signal.connect(self.on_gaze_data)
        self.video_thread.start()

        self.showFullScreen()

        QTimer.singleShot(2000, self.start_calibration)

    def resizeEvent(self, event):
        if hasattr(self, 'camera_label') and self.camera_label is not None:
            self.camera_label.move(self.width() - 340, 20)
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def start_calibration(self):
        self.calib_timer.start(50)
        self.terminal_display.append(
            "\n[SİSTEM]: Kalibrasyon başlıyor... Lütfen klavyenin ORTASINDAKİ kırmızı daireye bakın.\n")

    def on_gaze_data(self, dx, dy, ix, iy, q_image):
        self.last_ix = ix
        self.last_iy = iy

        # 2. Kamera Görüntüsünü UI'ya bas
        if hasattr(self, 'camera_label'):
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                self.camera_label.width(),
                self.camera_label.height(),
                Qt.KeepAspectRatio
            )
            self.camera_label.setPixmap(scaled_pixmap)

        if not self.keyboard.is_calibrating:
            self.keyboard.update_gaze_data(dx, dy)

    def collect_calibration_data(self):
        if hasattr(self, 'last_ix'):
            self.calibration_buffer.append((self.last_ix, self.last_iy))

            count = len(self.calibration_buffer)
            target_count = 60  # ~3 saniye

            progress = int((count / target_count) * 100)
            self.keyboard.calibration_progress = progress
            self.keyboard.update()

            if count >= target_count:
                self.finalize_calibration()

    def finalize_calibration(self):
        self.calib_timer.stop()

        data = np.array(self.calibration_buffer)
        if len(data) > 0:
            avg_x = np.mean(data[:, 0])
            avg_y = np.mean(data[:, 1])

            self.video_thread.set_calibration(avg_x, avg_y, 100.0)

            self.keyboard.is_calibrating = False
            self.keyboard.update()

            self.terminal_display.append("[SİSTEM]: Kalibrasyon Başarılı! Klavyeyi kullanabilirsiniz.\n")
            self.terminal_display.append("kullanici@goz-pc:~$ " + self.current_cmd)
        else:
            self.terminal_display.append("[HATA]: Kalibrasyon verisi alınamadı. Tekrar deneyin.\n")
            QTimer.singleShot(2000, self.start_calibration)

    def handle_key_input(self, key):
        if key.startswith("["): return  # Info mesajları

        if key == "SPACE":
            self.current_cmd += " "
        elif key == "BACK":
            self.current_cmd = self.current_cmd[:-1]
        elif key == "ENTER":
            self.execute_command()
            return
        elif key == "MODE":
            pass
        else:
            self.current_cmd += key

        self.update_terminal_line()

    def update_terminal_line(self):
        current_text = self.terminal_display.toPlainText()
        lines = current_text.split('\n')
        base_prompt = "kullanici@goz-pc:~$ "

        if len(lines) > 0 and base_prompt in lines[-1]:
            lines.pop()

        lines.append(base_prompt + self.current_cmd)
        self.terminal_display.setText("\n".join(lines))
        sb = self.terminal_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def execute_command(self):
        cmd = self.current_cmd
        self.terminal_display.append(f"\n> {cmd}\n")

        try:
            if cmd.startswith("cd "):
                self.terminal_display.append("Dizin değiştirildi (Simülasyon)\n")
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.stdout: self.terminal_display.append(result.stdout)
                if result.stderr: self.terminal_display.append(result.stderr)
        except Exception as e:
            self.terminal_display.append(f"Hata: {str(e)}")

        self.current_cmd = ""
        self.terminal_display.append("kullanici@goz-pc:~$ ")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
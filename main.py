import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pyautogui
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTextEdit, QLabel, QDesktopWidget
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush, QImage, QPixmap, QRegion

# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------
MOUSE_SENSITIVITY = 0.5
SMOOTHING_FACTOR = 0.5
MOUSE_DEADZONE = 8.0
BLINK_THRESHOLD = 0.30
CLICK_DELAY = 0.05

pyautogui.FAILSAFE = False

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    os.path.dirname(sys.modules["PyQt5"].__file__), "Qt", "plugins"
)


# ---------------------------------------------------------
# YARDIMCI SINIF: ÇIKIŞ BUTONU
# ---------------------------------------------------------
class ExitOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def set_progress(self, val):
        pass

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        p.setBrush(QColor(255, 0, 0, 200))  # Kırmızı
        p.setPen(Qt.NoPen)
        rect = self.rect()
        p.drawRoundedRect(rect, 20, 20)

        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Arial", 14, QFont.Bold))
        p.drawText(rect, Qt.AlignCenter, "ÇIKIŞ\nBÖLGESİ")


# ---------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
def calculate_ear(landmarks, indices):
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    dist_horizontal = np.linalg.norm(p1 - p2)
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    dist_vertical = np.linalg.norm(p3 - p4)
    if dist_horizontal == 0: return 0
    return dist_vertical / dist_horizontal


# ---------------------------------------------------------
# 1. VIDEO THREAD
# ---------------------------------------------------------
class VideoThread(QThread):
    gaze_signal = pyqtSignal(float, float, float, float, QImage)
    exit_mouse_mode_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.mouse_mode_active = False
        self.prev_dx = 0
        self.prev_dy = 0
        self.scroll_ref_y = None
        self.click_cooldown = 0
        self.emergency_blink_counter = 0
        self.calibrated_center = None
        self.eye_width_ref = 10.0
        self.BOX_SIZE = 350

    def run(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("KAMERA BAŞLATILDI")

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            dx, dy, ix, iy = 0, 0, 0, 0
            LEFT_EYE_IDX = [33, 133, 159, 145]
            RIGHT_EYE_IDX = [362, 263, 386, 374]
            NOSE_TIP_IDX = 1

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in landmarks])

                iris_left = mesh_points[468]
                ix, iy = iris_left[0], iris_left[1]

                left_eye_inner = mesh_points[33]
                left_eye_outer = mesh_points[133]
                current_eye_width = np.linalg.norm(left_eye_inner - left_eye_outer)

                if self.calibrated_center is not None:
                    scale_factor = current_eye_width / self.eye_width_ref
                    raw_dx = ix - self.calibrated_center[0]
                    raw_dy = iy - self.calibrated_center[1]
                    target_dx = raw_dx / scale_factor
                    target_dy = raw_dy / scale_factor
                    self.prev_dx = (self.prev_dx * (1 - SMOOTHING_FACTOR)) + (target_dx * SMOOTHING_FACTOR)
                    self.prev_dy = (self.prev_dy * (1 - SMOOTHING_FACTOR)) + (target_dy * SMOOTHING_FACTOR)
                    dx = self.prev_dx
                    dy = self.prev_dy

                ear_left = calculate_ear(landmarks, LEFT_EYE_IDX)
                ear_right = calculate_ear(landmarks, RIGHT_EYE_IDX)
                is_left_closed = ear_left < BLINK_THRESHOLD
                is_right_closed = ear_right < BLINK_THRESHOLD

                if self.mouse_mode_active:
                    cursor_x, cursor_y = pyautogui.position()

                    # ANINDA ÇIKIŞ
                    if cursor_x < self.BOX_SIZE and cursor_y < self.BOX_SIZE:
                        print("KUTUYA GİRİLDİ -> ANINDA ÇIKIŞ")
                        self.exit_mouse_mode_signal.emit()
                        time.sleep(1.5)
                        continue

                        # ACİL DURUM (Göz Kırpma)
                    if is_left_closed and is_right_closed:
                        self.emergency_blink_counter += 1
                        if self.emergency_blink_counter > 50:
                            print("ACİL GÖZ KIRPMA ÇIKIŞI!")
                            self.exit_mouse_mode_signal.emit()
                            self.emergency_blink_counter = 0
                            time.sleep(1.5)
                            continue
                    else:
                        self.emergency_blink_counter = 0

                    if not (is_left_closed and is_right_closed):
                        self.scroll_ref_y = None
                        if self.click_cooldown > 0:
                            self.click_cooldown -= 1
                        else:
                            if is_left_closed and not is_right_closed:
                                pyautogui.mouseDown(button='left')
                                time.sleep(CLICK_DELAY)
                                pyautogui.mouseUp(button='left')
                                self.click_cooldown = 20
                            elif is_right_closed and not is_left_closed:
                                pyautogui.mouseDown(button='right')
                                time.sleep(CLICK_DELAY)
                                pyautogui.mouseUp(button='right')
                                self.click_cooldown = 20

                        if not is_left_closed and not is_right_closed:
                            magnitude = math.sqrt(dx ** 2 + dy ** 2)
                            if magnitude > MOUSE_DEADZONE:
                                effective_mag = magnitude - MOUSE_DEADZONE
                                speed_curve = effective_mag ** 1.3
                                speed = (speed_curve * MOUSE_SENSITIVITY)
                                try:
                                    pyautogui.moveRel((dx / magnitude) * speed, (dy / magnitude) * speed)
                                except:
                                    pass

            final_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=final_rgb,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                                    circle_radius=1)
                )
            h, w, ch = final_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(final_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.gaze_signal.emit(dx, dy, float(ix), float(iy), qt_image)

        cap.release()

    def set_calibration(self, center_x, center_y, eye_w):
        self.calibrated_center = (center_x, center_y)
        self.eye_width_ref = eye_w
        self.prev_dx = 0
        self.prev_dy = 0

    def set_mouse_mode(self, active):
        self.mouse_mode_active = active
        self.emergency_blink_counter = 0


# ---------------------------------------------------------
# 2. RADYAL KLAVYE
# ---------------------------------------------------------
class RadialKeyboard(QWidget):
    key_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.hovered_key = None
        self.dwell_progress = 0

        base_keys = ["SPACE", "BACK", "ENTER", "MODE"]
        self.layouts = {
            "lowercase": list("abcdefghijklmnopqrstuvwxyz") + base_keys,
            "uppercase": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + base_keys,
            "numbers": list("0123456789+-*/=.,") + base_keys,
            "symbols": list("!@#$%^&()_[]{}:;<>?|\\~") + base_keys
        }

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
        if self.is_calibrating or self.magnitude < self.deadzone_radius:
            self.hovered_key = None
            self.dwell_progress = 0
            self.update()
            return
        cur = self.get_key_at_angle(self.angle)
        if cur == self.hovered_key:
            self.dwell_progress += 1
        else:
            self.hovered_key = cur
            self.dwell_progress = 0
        if self.dwell_progress >= self.dwell_max:
            self.trigger_selection()
            self.dwell_progress = 0
            self.hovered_key = None
        self.update()

    def get_key_at_angle(self, angle):
        d = angle
        if d < 0: d += 360
        if self.in_mode_select:
            if 0 <= d < 72:
                return "numbers"
            elif 72 <= d < 144:
                return "symbols"
            elif 144 <= d < 216:
                return "lowercase"
            elif 216 <= d < 288:
                return "uppercase"
            else:
                return "MOUSE"
        else:
            k = self.layouts[self.current_mode]
            step = 360 / len(k)
            idx = int(d / step)
            if idx >= len(k): idx = len(k) - 1
            return k[idx]

    def trigger_selection(self):
        if not self.hovered_key: return
        if self.hovered_key == "MODE":
            self.in_mode_select = True
            self.key_selected.emit("[MOD SEÇİM]")
            return
        if self.in_mode_select and self.hovered_key == "MOUSE":
            self.in_mode_select = False
            self.key_selected.emit("MOUSE")
            return
        if self.in_mode_select:
            self.current_mode = self.hovered_key
            self.in_mode_select = False
            self.key_selected.emit(f"[MOD: {self.current_mode}]")
            return
        self.key_selected.emit(self.hovered_key)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        c = QPointF(w / 2, h / 2)
        r = min(w, h) / 2 - 10
        if self.is_calibrating:
            p.setOpacity(0.5)
            p.setPen(QPen(QColor(255, 0, 0), 3))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 40, 40)
            p.setBrush(QColor(0, 255, 0))
            p.setPen(Qt.NoPen)
            p.drawEllipse(c, 40 * (self.calibration_progress / 100), 40 * (self.calibration_progress / 100))
        if self.in_mode_select:
            self.draw_mode_selector(p, c, r)
        else:
            self.draw_keys(p, c, r)
        p.setOpacity(1.0)
        p.setBrush(QColor(20, 20, 20))
        p.setPen(QPen(QColor(255, 255, 0), 2))
        p.drawEllipse(c, 30, 30)
        if not self.is_calibrating:
            cur_x = c.x() + (self.dx * 15)
            cur_y = c.y() + (self.dy * 15)
            p.setBrush(QColor(0, 255, 0))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cur_x, cur_y), 10, 10)

    def draw_keys(self, p, c, r):
        k = self.layouts[self.current_mode]
        step = 360 / len(k)
        for i, ky in enumerate(k):
            self.draw_pie_slice(p, c, r, -(i * step), -step, ky, ky == self.hovered_key)

    def draw_mode_selector(self, p, c, r):
        slices = [("SAYI", 0), ("SEM", 72), ("kçk", 144), ("BYK", 216), ("MOUSE", 288)]
        for txt, ang in slices:
            is_sel = (self.hovered_key == "numbers" and txt == "SAYI") or (
                    self.hovered_key == "symbols" and txt == "SEM") or (
                             self.hovered_key == "lowercase" and txt == "kçk") or (
                             self.hovered_key == "uppercase" and txt == "BYK") or (
                             self.hovered_key == "MOUSE" and txt == "MOUSE")
            self.draw_pie_slice(p, c, r, ang, 72, txt, is_sel)

    def draw_pie_slice(self, p, c, r, st, sp, txt, sel):
        if sel:
            p.setBrush(QColor(0, 180, 0))
        else:
            p.setBrush(QColor(20, 20, 20, 200))
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawPie(QRectF(c.x() - r, c.y() - r, r * 2, r * 2), int(st * 16), int(sp * 16))
        ang = math.radians(st + sp / 2)
        dist = r * 0.70
        tx = c.x() + dist * math.cos(ang)
        ty = c.y() - dist * math.sin(ang)
        p.setPen(Qt.white)
        p.setFont(QFont("Arial", 10, QFont.Bold))
        p.drawText(QRectF(tx - 30, ty - 15, 60, 30), Qt.AlignCenter, txt)


# ---------------------------------------------------------
# 3. ANA UYGULAMA
# ---------------------------------------------------------
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Göz Kontrol Paneli")

        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        self.term = QTextEdit(self)
        self.term.setReadOnly(True)
        self.term.setFocusPolicy(Qt.NoFocus)
        self.term.setStyleSheet(
            "background-color: rgba(0,0,0,150); color: #00FF00; font-family: 'Consolas'; font-size: 14px; border: 1px solid #00FF00;")
        self.term.setText(">> SİSTEM HAZIR\n>> Kalibrasyon bekleniyor...")
        self.term.show()

        self.cam = QLabel(self)
        self.cam.setStyleSheet("border: 2px solid #00FF00; background-color: black;")
        self.cam.show()

        self.kb = RadialKeyboard(self)
        self.kb.key_selected.connect(self.on_key)
        self.kb.show()

        self.exit_btn = ExitOverlay(self)
        self.exit_btn.resize(150, 150)
        self.exit_btn.hide()

        self.calib_data = []
        self.calib_timer = QTimer()
        self.calib_timer.timeout.connect(self.do_calib)

        self.thread = VideoThread()
        self.thread.gaze_signal.connect(self.on_gaze)
        self.thread.exit_mouse_mode_signal.connect(self.stop_mouse_mode)
        self.thread.start()

        # -----------------------------------------------
        # EKRAN SEÇİMİ: 1. EKRANA ZORLA (DEĞİŞİKLİK BURADA)
        # -----------------------------------------------
        monitor = QDesktopWidget().screenGeometry(0)  # 0 = Ana Ekran (1. Ekran)
        self.move(monitor.left(), monitor.top())  # Pencereyi 1. ekrana taşı
        self.showFullScreen()  # Sonra tam ekran yap

        QTimer.singleShot(2000, self.start_calib)

    def log(self, text):
        self.term.append(f">> {text}")
        self.term.verticalScrollBar().setValue(self.term.verticalScrollBar().maximum())

    def resizeEvent(self, e):
        w, h = self.width(), self.height()
        if self.thread.mouse_mode_active:
            self.exit_btn.setGeometry(0, 0, 150, 150)
        else:
            if w > 200:
                self.term.setGeometry(20, 20, 300, 150)
                self.cam.setGeometry(w - 260, 20, 240, 180)
                kb_size = 500
                kb_x = (w - kb_size) // 2
                kb_y = (h - kb_size) // 2 + 50
                self.kb.setGeometry(kb_x, kb_y, kb_size, kb_size)
            self.exit_btn.setGeometry(0, 0, 150, 150)
        super().resizeEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape: self.close()

    def start_calib(self):
        self.calib_timer.start(50)
        self.log("Kalibrasyon... Merkeze bakın.")

    def do_calib(self):
        if hasattr(self, 'cur_ix'):
            self.calib_data.append((self.cur_ix, self.cur_iy))
            prog = int((len(self.calib_data) / 60) * 100)
            self.kb.calibration_progress = prog
            self.kb.update()
            if len(self.calib_data) >= 60:
                self.calib_timer.stop()
                d = np.array(self.calib_data)
                self.thread.set_calibration(np.mean(d[:, 0]), np.mean(d[:, 1]), 100.0)
                self.kb.is_calibrating = False
                self.kb.update()
                self.log("Kalibrasyon Tamam.")

    def on_gaze(self, dx, dy, ix, iy, img):
        self.cur_ix, self.cur_iy = ix, iy
        if not self.thread.mouse_mode_active:
            if self.cam.isVisible(): self.cam.setPixmap(
                QPixmap.fromImage(img).scaled(self.cam.width(), self.cam.height(), Qt.KeepAspectRatio))
            if not self.kb.is_calibrating: self.kb.update_gaze_data(dx, dy)

    def on_key(self, k):
        if k == "MOUSE":
            self.start_mouse_mode()
            return
        if k.startswith("["):
            self.log(k)
            return
        try:
            if k == "ENTER":
                pyautogui.press('enter')
                self.log("ENTER")
            elif k == "BACK":
                pyautogui.press('backspace')
                self.log("SİL")
            elif k == "SPACE":
                pyautogui.press('space')
                self.log("BOŞLUK")
            elif k == "MODE":
                pass
            else:
                pyautogui.write(k)
                self.log(f"Yazıldı: {k}")
        except:
            pass

    def start_mouse_mode(self):
        self.thread.set_mouse_mode(True)
        self.kb.hide()
        self.term.hide()
        self.cam.hide()
        self.exit_btn.show()

        # MOUSE MODUNA GEÇERKEN DE EKRAN KONUMUNU GARANTİLE
        monitor = QDesktopWidget().screenGeometry(0)
        self.setGeometry(monitor.left(), monitor.top(), 150, 150)

        self.setMask(QRegion(0, 0, 150, 150))
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.exit_btn.set_progress(0)
        print("MOUSE MODU: AKTİF. Lütfen siyah terminal ekranını kontrol edin.")

    def stop_mouse_mode(self):
        self.thread.set_mouse_mode(False)
        self.exit_btn.hide()
        self.clearMask()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

        # 1. Ekranda tekrar tam ekran yap
        monitor = QDesktopWidget().screenGeometry(0)
        self.move(monitor.left(), monitor.top())
        self.showFullScreen()

        self.raise_()
        self.activateWindow()

        # Bileşenleri geri getir
        self.kb.show()
        self.term.show()
        self.cam.show()
        self.log("Klavye Moduna Dönüldü.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
import sys
import numpy as np
import sounddevice as sd
import cv2
from scipy.signal import butter, lfilter
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QFrame, QStackedLayout, QMainWindow)
from PyQt5.QtCore import Qt, QTimer, QRect, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import threading

# --- CONFIGURATION ---
SAMPLE_RATE = 48000  # UMA-16 default
BLOCK_SIZE = 4096    # Audio block size (maior precisão para voz)
N_MICS = 16          # UMA-16
CAM_INDEX = 0        # Default webcam
AUDIO_DEVICE = None  # Set to None for default, or use sd.query_devices() to find UMA-16 index
MIC_SPACING = 0.01   # 1 cm spacing (meters)
SOUND_SPEED = 343.0  # Speed of sound in air (m/s)

# Beamforming parameters
N_ANGLES = 90  # Number of look directions (for heatmap)
angles = np.linspace(-np.pi/2, np.pi/2, N_ANGLES)  # -90 to +90 degrees

# Posições dos microfones em grid 4x4 (centralizado no zero, câmera no centro)
N_ROWS = 4
N_COLS = 4
mic_positions = np.zeros((N_MICS, 3))
for idx in range(N_MICS):
    row = idx // N_COLS
    col = idx % N_COLS
    # Centraliza o grid em (0,0)
    x = (col - (N_COLS-1)/2) * MIC_SPACING
    y = ((N_ROWS-1)/2 - row) * MIC_SPACING  # y decresce de cima para baixo
    mic_positions[idx, 0] = x
    mic_positions[idx, 1] = y

# --- High-pass and Low-pass filters ---
def butter_highpass(cutoff, fs, order=5):
    """Cria um filtro passa-alta Butterworth."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    """Aplica filtro passa-alta."""
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data, axis=-1)

def butter_lowpass(cutoff, fs, order=5):
    """Cria um filtro passa-baixa Butterworth."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    """Aplica filtro passa-baixa."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, data, axis=-1)

# --- AUDIO CALLBACK ---
audio_buffer = np.zeros((N_MICS, BLOCK_SIZE))
def audio_callback(indata, frames, time, status):
    """Callback chamado a cada bloco de áudio capturado."""
    global audio_buffer
    if status:
        print(status)
    # indata shape: (frames, channels)
    audio_buffer = indata.T

# --- DELAY-AND-SUM BEAMFORMER ---
def delay_and_sum(audio, mic_positions, angles, fs, sound_speed):
    """
    audio: shape (N_MICS, BLOCK_SIZE)
    mic_positions: (N_MICS, 3)
    angles: array of look directions (radians)
    fs: sample rate
    sound_speed: speed of sound
    Returns: intensity per angle (N_ANGLES,)
    """
    n_mics, n_samples = audio.shape
    center = np.mean(mic_positions[:, 0])
    # Assume source is at z=1m, y=0
    z = 1.0
    y = 0.0
    intensities = np.zeros(len(angles))
    for i, theta in enumerate(angles):
        # Calculate delays for each mic
        # Source at (x=0, y=0, z=1), look direction theta
        look_vec = np.array([np.sin(theta), 0, np.cos(theta)])
        rel_pos = mic_positions - np.array([center, 0, 0])
        delays = (rel_pos @ look_vec) / sound_speed  # in seconds
        sample_delays = (delays * fs).astype(int)
        # Align and sum
        aligned = np.zeros(n_samples)
        for m in range(n_mics):
            d = sample_delays[m]
            if d >= 0:
                aligned[d:] += audio[m, :n_samples-d]
            else:
                aligned[:d] += audio[m, -d:]
        # Intensity = RMS
        intensities[i] = np.sqrt(np.mean(aligned**2))
    return intensities

class EditableLabel(QWidget):
    def __init__(self, value, minval, maxval, suffix, label_text, on_change, parent=None):
        super().__init__(parent)
        self.value = value
        self.minval = minval
        self.maxval = maxval
        self.suffix = suffix
        self.label_text = label_text
        self.on_change = on_change
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.label = QLabel(f"{self.label_text}: {self.value} {self.suffix}")
        self.label.setStyleSheet("color: #fff; font-size: 20px; font-weight: 600; padding: 2px 8px; border-radius: 8px;")
        self.label.mousePressEvent = self.activate_edit
        self.edit = QLineEdit(str(self.value))
        self.edit.setStyleSheet("background: #222; color: #fff; font-size: 20px; border-radius: 8px; padding: 2px 8px;")
        self.edit.hide()
        self.edit.returnPressed.connect(self.finish_edit)
        self.edit.editingFinished.connect(self.finish_edit)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit)
        self.setLayout(self.layout)
    def activate_edit(self, event):
        """Mostra o campo de input ao clicar no label."""
        self.label.hide()
        self.edit.setText(str(self.value))
        self.edit.show()
        self.edit.setFocus()
        self.edit.selectAll()
    def finish_edit(self):
        """Atualiza o valor ao sair do campo de input."""
        try:
            val = int(self.edit.text())
            if val < self.minval:
                val = self.minval
            if val > self.maxval:
                val = self.maxval
            self.value = val
            self.label.setText(f"{self.label_text}: {self.value} {self.suffix}")
            self.on_change(self.value)
        except ValueError:
            pass
        self.edit.hide()
        self.label.show()
    def set_value(self, value):
        self.value = value
        self.label.setText(f"{self.label_text}: {self.value} {self.suffix}")

class MplCanvas(FigureCanvas):
    def __init__(self, width=4, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="none")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, facecolor="none")
        self.setStyleSheet("background: transparent;")
        self.setMinimumHeight(180)

class CameraWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border-radius: 18px;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Câmara")
        self.fullscreen = False
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.fullscreen_btn = QPushButton("⛶", self)
        self.fullscreen_btn.setFixedSize(36, 36)
        self.fullscreen_btn.setStyleSheet("background: rgba(30,30,30,180); color: #fff; border: none; border-radius: 18px; font-size: 20px;")
        self.fullscreen_btn.clicked.connect(self._call_parent_fullscreen)
        self.fullscreen_btn.raise_()
        # Label para coordenadas e RGB
        self.coord_label = QLabel(self)
        self.coord_label.setStyleSheet("background: rgba(30,30,30,180); color: #fff; border-radius: 8px; font-size: 13px; padding: 4px 10px;")
        self.coord_label.move(16, self.height() - 36)
        self.coord_label.hide()
        self.rgb_image = None
        self.setMouseTracking(True)
    def set_rgb_image(self, rgb_image):
        """Atualiza a imagem RGB para leitura de pixel."""
        self.rgb_image = rgb_image
    def mouseMoveEvent(self, event):
        """Mostra coordenadas e RGB do pixel sob o mouse."""
        if self.rgb_image is not None:
            x = event.x()
            y = event.y()
            pm = self.pixmap()
            if pm is not None:
                # Ajusta para a área centralizada
                w, h = pm.width(), pm.height()
                x0 = (self.width() - w) // 2
                y0 = (self.height() - h) // 2
                if x0 <= x < x0 + w and y0 <= y < y0 + h:
                    px = int((x - x0) * self.rgb_image.shape[1] / w)
                    py = int((y - y0) * self.rgb_image.shape[0] / h)
                    if 0 <= px < self.rgb_image.shape[1] and 0 <= py < self.rgb_image.shape[0]:
                        rgb = self.rgb_image[py, px]
                        self.coord_label.setText(f"({px}, {py})  RGB: {tuple(rgb)}")
                        self.coord_label.move(16, self.height() - 36)
                        self.coord_label.show()
                        return
        self.coord_label.hide()
    def leaveEvent(self, event):
        self.coord_label.hide()
        super().leaveEvent(event)
    def _call_parent_fullscreen(self):
        if hasattr(self.parent(), 'toggle_camera_fullscreen'):
            self.parent().toggle_camera_fullscreen()

class FullScreenCameraWidget(QWidget):
    def __init__(self, camera_pixmap_func, exit_fullscreen_callback, parent=None):
        super().__init__(parent)
        self.camera_pixmap_func = camera_pixmap_func
        self.exit_fullscreen_callback = exit_fullscreen_callback
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.fullscreen_btn = QPushButton("⤫", self)
        self.fullscreen_btn.setFixedSize(36, 36)
        self.fullscreen_btn.setStyleSheet("background: rgba(30,30,30,180); color: #fff; border: none; border-radius: 18px; font-size: 20px;")
        self.fullscreen_btn.clicked.connect(self.exit_fullscreen_callback)
        self.fullscreen_btn.raise_()
        self.resizeEvent(None)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Fundo cinza translúcido
        painter.fillRect(self.rect(), QColor(30,30,30,180))
        # Desenhar a imagem da câmera centralizada
        pm = self.camera_pixmap_func()
        if pm is not None:
            widget_ratio = self.width() / self.height()
            pm_ratio = pm.width() / pm.height()
            if widget_ratio > pm_ratio:
                # Limita pela altura
                h = self.height()
                w = int(h * pm_ratio)
            else:
                w = self.width()
                h = int(w / pm_ratio)
            x = (self.width() - w) // 2
            y = (self.height() - h) // 2
            painter.drawPixmap(x, y, w, h, pm)
    def resizeEvent(self, event):
        # Botão no canto inferior direito da área da câmera
        btn_size = self.fullscreen_btn.size()
        margin = 32
        x = self.width() - btn_size.width() - margin
        y = self.height() - btn_size.height() - margin
        self.fullscreen_btn.move(x, y)
        if event:
            super().resizeEvent(event)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEAK FINDER - Acoustic Camera")
        self.setStyleSheet("background-color: #181a20; color: #fff; font-family: 'Segoe UI', Arial;")
        self.setFont(QFont("Segoe UI", 11))
        self.hp_value = 80    # Filtro passa-alta padrão para voz
        self.lp_value = 3000 # Filtro passa-baixa padrão para voz
        self.intensity_history = [0]*100
        self.heatmap_data = np.zeros(N_ANGLES)
        self.fullscreen_camera = False
        self.fullscreen_camera_widget = None
        self.paused = False
        self.saved_frame_count = 0
        self.last_frame = None
        self.frame_counter = 0
        self.update_graphics_every = 8  # Atualiza gráficos/processamento a cada 8 frames
        self.smooth_intensity = 0.0
        self.smooth_idx = 0.0
        self.spot_params = None  # (x_pos, y_pos, radius, color_map)
        self.alpha = 0.7
        self.airleak_mode = False
        self.prev_hp_value = self.hp_value
        self.prev_lp_value = self.lp_value
        # --- Inicializa a câmera ANTES da thread ---
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_lock = threading.Lock()
        self.stop_video_thread = False
        self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
        self.video_thread.start()
        self.init_ui()
        # Timer para vídeo (máxima fluidez)
        self.video_timer = QTimer()
        self.video_timer.setTimerType(Qt.PreciseTimer)
        self.video_timer.timeout.connect(self.update_video)
        self.video_timer.start(5)  # Intervalo menor para maior fluidez
        # Timer para gráficos/processamento
        self.proc_timer = QTimer()
        self.proc_timer.setTimerType(Qt.PreciseTimer)
        self.proc_timer.timeout.connect(self.update_processing)
        self.proc_timer.start(200)
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=N_MICS,
            device=AUDIO_DEVICE,
            callback=audio_callback
        )
        self.stream.start()

    def init_ui(self):
        # Camera area
        self.camera_label = CameraWidget(self)
        self.camera_label.setMinimumSize(400, 300)
        # High-pass
        self.hp_edit = EditableLabel(self.hp_value, 20, 10000, "Hz", "High-pass", self.on_hp_change)
        self.hp_slider = QSlider(Qt.Horizontal)
        self.hp_slider.setMinimum(20)
        self.hp_slider.setMaximum(10000)
        self.hp_slider.setValue(self.hp_value)
        self.hp_slider.setStyleSheet(self.slider_style())
        self.hp_slider.valueChanged.connect(self.on_hp_change)
        # Low-pass
        self.lp_edit = EditableLabel(self.lp_value, 1000, 20000, "Hz", "Low-pass", self.on_lp_change)
        self.lp_slider = QSlider(Qt.Horizontal)
        self.lp_slider.setMinimum(1000)
        self.lp_slider.setMaximum(20000)
        self.lp_slider.setValue(self.lp_value)
        self.lp_slider.setStyleSheet(self.slider_style())
        self.lp_slider.valueChanged.connect(self.on_lp_change)
        # Label para coordenadas e RGB
        self.coord_info_label = QLabel("Passe o mouse sobre a câmera para ver as coordenadas e RGB.")
        self.coord_info_label.setStyleSheet("background: rgba(30,30,30,180); color: #fff; border-radius: 8px; font-size: 15px; padding: 8px 18px;")
        # Gráficos
        self.intensity_canvas = MplCanvas(width=3.5, height=1.5)
        self.heatmap_canvas = MplCanvas(width=3.5, height=1.5)
        # Opções (botões)
        self.save_btn = QPushButton("Salvar Frame")
        self.save_btn.setStyleSheet(self.button_style())
        self.save_btn.clicked.connect(self.save_frame)
        self.pause_btn = QPushButton("Pausar Vídeo")
        self.pause_btn.setStyleSheet(self.button_style())
        self.pause_btn.clicked.connect(self.toggle_pause)
        # --- Airleak Mode Button ---
        self.airleak_btn = QPushButton("Airleak Mode")
        self.airleak_btn.setStyleSheet(self.button_style())
        self.airleak_btn.clicked.connect(self.toggle_airleak_mode)
        options_box = QHBoxLayout()
        options_box.addWidget(self.save_btn)
        options_box.addWidget(self.pause_btn)
        options_box.addWidget(self.airleak_btn)
        options_widget = QWidget()
        options_widget.setLayout(options_box)
        options_widget.setStyleSheet(self.card_style())
        # Layouts responsivos
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(18)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        # Top: câmera e gráficos
        self.top_layout = QHBoxLayout()
        cam_box = QVBoxLayout()
        cam_box.addWidget(self.camera_label, stretch=3)
        cam_box.setContentsMargins(0,0,0,0)
        self.top_layout.addLayout(cam_box, stretch=3)
        graph_box = QVBoxLayout()
        graph_box.addWidget(self.intensity_canvas, stretch=1)
        graph_box.addWidget(self.heatmap_canvas, stretch=1)
        self.top_layout.addLayout(graph_box, stretch=2)
        self.main_layout.addLayout(self.top_layout, stretch=4)
        # Meio: sliders
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(self.hp_edit)
        self.main_layout.addWidget(self.hp_slider)
        self.main_layout.addWidget(self.lp_edit)
        self.main_layout.addWidget(self.lp_slider)
        # Label de coordenadas e RGB
        self.main_layout.addWidget(self.coord_info_label)
        # Opções (botões)
        self.main_layout.addWidget(options_widget, stretch=1)
        self.setLayout(self.main_layout)

    def slider_style(self):
        return """
        QSlider::groove:horizontal {height: 8px; background: #2d313a; border-radius: 4px;}
        QSlider::handle:horizontal {background: #fff; border: 2px solid #888; width: 18px; margin: -6px 0; border-radius: 9px;}
        QSlider::sub-page:horizontal {background: #00e676; border-radius: 4px;}
        """
    def button_style(self):
        return "background: #23272f; color: #fff; border: none; border-radius: 10px; padding: 10px 18px; font-size: 15px; font-weight: 600;"
    def card_style(self):
        return "background: #23272f; color: #fff; border-radius: 18px; min-height: 120px; min-width: 220px; font-size: 16px; padding: 18px;"

    def on_hp_change(self, value):
        """Callback para mudança do high-pass."""
        if isinstance(value, int):
            self.hp_value = value
            self.hp_slider.setValue(value)
            self.hp_edit.set_value(value)
        else:
            self.hp_value = self.hp_slider.value()
            self.hp_edit.set_value(self.hp_value)
    def on_lp_change(self, value):
        """Callback para mudança do low-pass."""
        if isinstance(value, int):
            self.lp_value = value
            self.lp_slider.setValue(value)
            self.lp_edit.set_value(value)
        else:
            self.lp_value = self.lp_slider.value()
            self.lp_edit.set_value(self.lp_value)

    def video_capture_loop(self):
        """Thread para capturar frames da câmera continuamente."""
        while not self.stop_video_thread:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self.frame_lock:
                    self.last_frame = frame.copy()
            # Pequeno sleep para não travar a CPU
            cv2.waitKey(1)

    def update_video(self):
        """Atualiza apenas o vídeo da câmera, desenhando a bolinha do beamforming."""
        if self.paused:
            with self.frame_lock:
                frame = self.last_frame.copy() if self.last_frame is not None else None
            if frame is None:
                return
        else:
            with self.frame_lock:
                frame = self.last_frame.copy() if self.last_frame is not None else None
            if frame is None:
                return
        # Desenhar spot cinza translúcido usando overlay e contorno preto
        if self.spot_params is not None:
            x_pos, y_pos, radius, spot_color = self.spot_params
            overlay = frame.copy()
            cv2.circle(overlay, (x_pos, y_pos), radius, spot_color[:3], -1)
            # Contorno preto
            cv2.circle(overlay, (x_pos, y_pos), radius, (0,0,0), 4)
            alpha = spot_color[3] / 255.0
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.camera_label.set_rgb_image(rgb_image)
        # Atualiza label de coordenadas e RGB
        if self.camera_label.coord_label.isVisible():
            self.coord_info_label.setText(self.camera_label.coord_label.text())
        else:
            self.coord_info_label.setText("Passe o mouse sobre a câmera para ver as coordenadas e RGB.")
        # Atualiza tela cheia se estiver ativa
        if self.fullscreen_camera and self.fullscreen_camera_widget:
            self.fullscreen_camera_widget.update()

    def update_processing(self):
        """Processamento pesado: áudio, beamforming e gráficos."""
        if self.paused:
            with self.frame_lock:
                frame = self.last_frame.copy() if self.last_frame is not None else None
            if frame is None:
                return
        else:
            with self.frame_lock:
                frame = self.last_frame.copy() if self.last_frame is not None else None
            if frame is None:
                return
        # Processamento de áudio
        hp_cut = self.hp_value
        lp_cut = self.lp_value
        if lp_cut <= hp_cut:
            lp_cut = hp_cut + 1
        audio = audio_buffer.copy()
        audio = highpass_filter(audio, hp_cut, SAMPLE_RATE, 4)
        audio = lowpass_filter(audio, lp_cut, SAMPLE_RATE, 4)
        # Aplica janela de Hanning para reduzir vazamento espectral
        window = np.hanning(audio.shape[1])
        audio = audio * window
        intensities = delay_and_sum(audio, mic_positions, angles, SAMPLE_RATE, SOUND_SPEED)
        abs_max_intensity = np.max(intensities)
        max_idx = np.argmax(intensities)
        self.smooth_intensity = self.alpha * abs_max_intensity + (1 - self.alpha) * self.smooth_intensity
        self.smooth_idx = self.alpha * max_idx + (1 - self.alpha) * self.smooth_idx
        # --- Spot pelo centro de massa das intensidades dos microfones (2D) ---
        h, w, _ = frame.shape
        mic_energies = np.sqrt(np.mean(audio**2, axis=1))
        # Reordena as energias conforme o mapeamento físico do usuário
        mic_map = np.array([
            8, 7, 10, 9,
            6, 5, 12, 11,
            4, 3, 14, 13,
            2, 1, 16, 15
        ]) - 1  # zero-based
        mic_energies_ordered = mic_energies[mic_map]
        mic_coords = np.array([(x, y) for y in range(4) for x in range(4)])
        total_energy = np.sum(mic_energies_ordered)
        # --- Threshold adaptativo para detecção real de som ---
        if not hasattr(self, 'background_level'):
            self.background_level = 0.0
        self.background_level = 0.98 * getattr(self, 'background_level', 0.0) + 0.02 * abs_max_intensity
        detection_threshold = self.background_level * 1.5 + 1e-6  # 50% acima do fundo
        detected = abs_max_intensity > detection_threshold
        # Suavização exponencial da posição do spot
        if not hasattr(self, 'smooth_x_spot'):
            self.smooth_x_spot = 1.5
            self.smooth_y_spot = 1.5
        alpha_spot = 0.7  # 0.0 = sem suavização, 1.0 = máximo suavização
        if detected and total_energy > 0:
            x_spot = np.sum(mic_coords[:, 0] * mic_energies_ordered) / total_energy
            y_spot = np.sum(mic_coords[:, 1] * mic_energies_ordered) / total_energy
            self.smooth_x_spot = alpha_spot * x_spot + (1 - alpha_spot) * self.smooth_x_spot
            self.smooth_y_spot = alpha_spot * y_spot + (1 - alpha_spot) * self.smooth_y_spot
            x_img = int(round(self.smooth_x_spot * (w - 1) / 3.0))
            y_img = int(round(self.smooth_y_spot * (h - 1) / 3.0))
            # Estilização: cor e alpha baseados na intensidade
            norm_int = np.clip(self.smooth_intensity / 0.05, 0, 1)
            color_map = cv2.applyColorMap(np.array([int(np.clip(norm_int * 255, 0, 255))], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
            radius = int(30 + 30 * norm_int)
            spot_alpha = 0.5 * norm_int + 0.2
            # Desenhar spot com alpha
            overlay = frame.copy()
            cv2.circle(overlay, (x_img, y_img), radius, color_map, -1)
            cv2.circle(overlay, (x_img, y_img), radius, (0,0,0), 4)  # contorno preto
            frame = cv2.addWeighted(overlay, spot_alpha, frame, 1 - spot_alpha, 0)
            self.spot_params = (x_img, y_img, radius, tuple(color_map)+(int(255*spot_alpha),))
        else:
            self.spot_params = None
        # Atualiza gráfico de intensidade
        self.intensity_history.append(self.smooth_intensity)
        if len(self.intensity_history) > 100:
            self.intensity_history.pop(0)
        self.intensity_canvas.ax.clear()
        self.intensity_canvas.ax.plot(self.intensity_history, color="#00e676", linewidth=2)
        self.intensity_canvas.ax.set_facecolor("none")
        self.intensity_canvas.ax.set_title("Intensidade do Som", color="#fff", fontsize=12, pad=24, loc='center')
        self.intensity_canvas.ax.tick_params(axis='x', colors='#fff')
        self.intensity_canvas.ax.tick_params(axis='y', colors='#fff')
        self.intensity_canvas.ax.spines['bottom'].set_color('#fff')
        self.intensity_canvas.ax.spines['top'].set_color('#fff')
        self.intensity_canvas.ax.spines['right'].set_color('#fff')
        self.intensity_canvas.ax.spines['left'].set_color('#fff')
        self.intensity_canvas.ax.set_ylim(0, max(0.05, max(self.intensity_history)))
        self.intensity_canvas.draw()
        # Atualiza heatmap
        self.heatmap_canvas.ax.clear()
        self.heatmap_canvas.ax.imshow([intensities], aspect='auto', cmap='jet', extent=[-90, 90, 0, 1])
        self.heatmap_canvas.ax.set_facecolor("none")
        self.heatmap_canvas.ax.set_title("Heatmap Beamforming", color="#fff", fontsize=12, pad=24, loc='center')
        self.heatmap_canvas.ax.get_yaxis().set_visible(False)
        self.heatmap_canvas.ax.tick_params(axis='x', colors='#fff')
        self.heatmap_canvas.ax.spines['bottom'].set_color('#fff')
        self.heatmap_canvas.ax.spines['top'].set_color('#fff')
        self.heatmap_canvas.ax.spines['right'].set_color('#fff')
        self.heatmap_canvas.ax.spines['left'].set_color('#fff')
        self.heatmap_canvas.draw()
        print("Energias dos mics:", mic_energies)

    def toggle_camera_fullscreen(self):
        """Alterna entre modo tela cheia da câmera e modo normal."""
        if not self.fullscreen_camera:
            self.fullscreen_camera = True
            # Cria widget de tela cheia customizado
            def get_camera_pixmap():
                return self.camera_label.pixmap()
            self.fullscreen_camera_widget = FullScreenCameraWidget(get_camera_pixmap, self.toggle_camera_fullscreen)
            self.fullscreen_camera_widget.showFullScreen()
        else:
            self.fullscreen_camera = False
            if self.fullscreen_camera_widget:
                self.fullscreen_camera_widget.close()
                self.fullscreen_camera_widget = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # O botão já é reposicionado pelo CameraWidget

    def closeEvent(self, event):
        self.stop_video_thread = True
        if self.video_thread.is_alive():
            self.video_thread.join(timeout=1)
        self.cap.release()
        self.stream.stop()
        event.accept()

    def save_frame(self):
        """Salva o último frame exibido como PNG."""
        if self.last_frame is not None:
            filename = f"frame_{self.saved_frame_count}.png"
            cv2.imwrite(filename, self.last_frame)
            self.saved_frame_count += 1
    def toggle_pause(self):
        """Pausa ou retoma o vídeo da câmera."""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.setText("Retomar Vídeo")
        else:
            self.pause_btn.setText("Pausar Vídeo")

    def toggle_airleak_mode(self):
        """Toggle Airleak Mode: sets HP to 20kHz, LP to 40kHz, updates UI, and visually indicates mode."""
        if not self.airleak_mode:
            # Save previous values
            self.prev_hp_value = self.hp_value
            self.prev_lp_value = self.lp_value
            # Set to airleak values
            self.hp_value = 20000
            self.lp_value = 40000
            self.hp_slider.setValue(self.hp_value)
            self.lp_slider.setValue(self.lp_value)
            self.hp_edit.set_value(self.hp_value)
            self.lp_edit.set_value(self.lp_value)
            self.airleak_btn.setText("Airleak Mode ON")
            self.airleak_btn.setStyleSheet("background: #00e676; color: #181a20; border: none; border-radius: 10px; padding: 10px 18px; font-size: 15px; font-weight: 600;")
            self.airleak_mode = True
        else:
            # Restore previous values
            self.hp_value = self.prev_hp_value
            self.lp_value = self.prev_lp_value
            self.hp_slider.setValue(self.hp_value)
            self.lp_slider.setValue(self.lp_value)
            self.hp_edit.set_value(self.hp_value)
            self.lp_edit.set_value(self.lp_value)
            self.airleak_btn.setText("Airleak Mode")
            self.airleak_btn.setStyleSheet(self.button_style())
            self.airleak_mode = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 750)
    window.show()
    sys.exit(app.exec_()) 
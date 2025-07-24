# Acoustic Camera 2D - Final Version
#
# This script uses a 16-channel UMA-16 microphone array and a webcam
# to create a real-time visualization of sound sources. It performs 2D
# beamforming to locate sounds horizontally and vertically.

import sounddevice as sd
import numpy as np
from scipy.signal import butter, sosfilt
import cv2
import time
import tkinter as tk
from tkinter import ttk, Frame, Label, Button, Entry, Scale, HORIZONTAL
from PIL import Image, ImageTk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import glob

# --- 1. CONFIGURATION & TUNING ---
# Audio Settings
DEVICE_NAME = 'UMA16'      # Part of the device name to search for
SAMPLE_RATE = 48000        # UMA-16 sample rate
CHANNELS_TO_USE = 16       # Use all 16 channels for 2D processing
BLOCK_DURATION = 100       # ms. Increase if your PC is slow and still freezing
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION / 1000)

# --- IMPORTANT TUNING PARAMETERS ---
# For testing with voice/snaps, use a wide range like [800, 16000]
# For actual air leaks, use a high range like [15000, 22000]
FREQUENCY_RANGE = [4995, 5005]

# Video Settings
CAMERA_ID = 0              # Camera index. 0 is usually the default.
CAMERA_FOV_H = 65.0        # HORIZONTAL Field of View (degrees). Tune for accuracy.
CAMERA_FOV_V = 40.0        # VERTICAL Field of View (degrees). Tune for accuracy.

# --- 2D Microphone Array Geometry (UMA-16) ---
# Defines the (x, y) position of each microphone in meters.
# IMPORTANT: The order of mic_positions must match the channel order from the device.
# If the beamforming is mirrored or inaccurate, try flipping the sign of x or y below.
mic_positions = np.array([
    [-0.066,  0.066], [-0.022,  0.066], [0.022,  0.066], [0.066,  0.066],   # Row 1: MIC8, MIC7, MIC10, MIC9
    [-0.066,  0.022], [-0.022,  0.022], [0.022,  0.022], [0.066,  0.022],   # Row 2: MIC6, MIC5, MIC12, MIC11
    [-0.066, -0.022], [-0.022, -0.022], [0.022, -0.022], [0.066, -0.022],   # Row 3: MIC4, MIC3, MIC14, MIC13
    [-0.066, -0.066], [-0.022, -0.066], [0.022, -0.066], [0.066, -0.066],   # Row 4: MIC2, MIC1, MIC16, MIC15
])

# --- MIRRORING OPTIONS FOR TROUBLESHOOTING ---
# If the beamforming is mirrored horizontally, uncomment the next line:
# mic_positions[:, 0] *= -1
# If the beamforming is mirrored vertically, uncomment the next line:
# mic_positions[:, 1] *= -1

# Processing & Visualization Settings
# Coarse resolution for real-time performance. Decrease step for more precision.
H_SCAN_ANGLES = (-70, 71, 9)  # Scan horizontally in 15-degree steps
V_SCAN_ANGLES = (-45, 46, 9)  # Scan vertically in 15-degree steps
SPEED_OF_SOUND = 343.0         # m/s
HOTSPOT_ALPHA = 0.5            # Transparency of the hotspot circle

# --- Global variable to share data between threads ---
latest_power_map = None
smoothed_power_map = None
SMOOTHING_FACTOR = 0.8  # Between 0 (no smoothing) and 1 (very smooth)

# UI Theme settings
DARK_MODE = False

# --- Helper Functions ---
def find_device_id():
    """Finds the UMA-16 device ID by searching for its name."""
    print("Searching for audio devices...")
    for i, device in enumerate(sd.query_devices()):
        if DEVICE_NAME.lower() in device['name'].lower():
            print(f"✅ Found device: ID={i}, Name='{device['name']}'")
            if device['max_input_channels'] >= CHANNELS_TO_USE:
                return i
    return None

def create_bandpass_filter(low_hz, high_hz, fs):
    """Creates a digital bandpass filter."""
    nyquist = 0.5 * fs
    low = low_hz / nyquist
    high = high_hz / nyquist
    sos = butter(8, [low, high], btype='band', output='sos')  # Increased from 5 to 8
    return sos

# --- Core 2D Audio Processing Callback ---
def audio_callback(indata, frames, time, status):
    """This function is called by sounddevice in a separate thread. Now uses frequency-domain beamforming for precise frequency selectivity."""
    global latest_power_map
    if status:
        print(f"Audio Status: {status}")

    # FFT parameters
    N = indata.shape[0]  # Number of samples in this block
    fft_data = np.fft.rfft(indata, axis=0)
    freqs = np.fft.rfftfreq(N, d=1/SAMPLE_RATE)

    # Find the closest FFT bin to the center frequency
    center_freq = (FREQUENCY_RANGE[0] + FREQUENCY_RANGE[1]) / 2
    freq_bin = np.argmin(np.abs(freqs - center_freq))

    # Extract the FFT value at the target frequency for each channel
    fft_at_freq = fft_data[freq_bin, :]  # shape: (CHANNELS_TO_USE,)

    h_angles = np.deg2rad(np.arange(*H_SCAN_ANGLES))
    v_angles = np.deg2rad(np.arange(*V_SCAN_ANGLES))
    power_map = np.zeros((len(v_angles), len(h_angles)))

    wavelength = SPEED_OF_SOUND / center_freq
    k = 2 * np.pi / wavelength  # Wavenumber

    # --- Nested loop for 2D scanning ---
    for v_idx, v_angle in enumerate(v_angles):
        for h_idx, h_angle in enumerate(h_angles):
            # Calculate a direction vector for the current scan angle
            dir_vector = np.array([np.cos(v_angle) * np.sin(h_angle), np.sin(v_angle)])
            # Calculate phase shifts for each mic (steering vector)
            phase_shifts = np.exp(-1j * k * np.dot(mic_positions, dir_vector))
            # Apply steering vector and sum across channels
            beamformed = np.sum(fft_at_freq * phase_shifts)
            power_map[v_idx, h_idx] = np.abs(beamformed) ** 2

    # Remove background offset
    power_map = power_map - np.min(power_map)

    # Normalize
    if np.max(power_map) > 0:
        power_map = power_map / np.max(power_map)

    # Apply a threshold to suppress noise
    threshold = 0.2  # Try values between 0.1 and 0.3
    power_map[power_map < threshold] = 0

    # --- Only keep the main hotspot (global maximum) ---
    if np.max(power_map) > 0:
        max_idx = np.unravel_index(np.argmax(power_map), power_map.shape)
        mask = np.zeros_like(power_map)
        mask[max_idx] = power_map[max_idx]
        power_map = mask

    global smoothed_power_map
    if smoothed_power_map is None:
        smoothed_power_map = power_map
    else:
        smoothed_power_map = SMOOTHING_FACTOR * smoothed_power_map + (1 - SMOOTHING_FACTOR) * power_map

    latest_power_map = smoothed_power_map

class MicBarGraphPopup(tk.Toplevel):
    def __init__(self, parent, get_intensities_func):
        super().__init__(parent)
        self.title('Microphone Intensities')
        self.configure(bg='#23272f')
        self.geometry('520x350')
        self.get_intensities_func = get_intensities_func
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.closed = False
        self.fig, self.ax = plt.subplots(figsize=(6,3), facecolor='#23272f')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.after(200, self.update_plot)
    def update_plot(self):
        if self.closed:
            return
        intensities = self.get_intensities_func()
        # Normalize for better visual difference
        if np.max(intensities) > 0:
            norm_intensities = intensities / np.max(intensities)
        else:
            norm_intensities = intensities
        self.ax.clear()
        self.ax.bar(np.arange(1, 17), norm_intensities, color='#00e676')
        self.ax.set_xlabel('Microphone', color='#fff')
        self.ax.set_ylabel('Normalized Intensity', color='#fff')
        self.ax.set_title('Microphone Intensities', color='#fff')
        self.ax.set_facecolor('#23272f')
        self.ax.tick_params(axis='x', colors='#fff')
        self.ax.tick_params(axis='y', colors='#fff')
        self.ax.spines['bottom'].set_color('#fff')
        self.ax.spines['top'].set_color('#fff')
        self.ax.spines['right'].set_color('#fff')
        self.ax.spines['left'].set_color('#fff')
        self.ax.set_ylim(0, 1.05)
        self.canvas.draw()
        self.after(200, self.update_plot)
    def close(self):
        self.closed = True
        self.destroy()

class FramesPopup(tk.Toplevel):
    def __init__(self, parent, image_pattern='acoustic_image_*.png'):
        super().__init__(parent)
        self.title('Saved Frames')
        self.configure(bg='#23272f')
        self.geometry('800x600')
        self.image_pattern = image_pattern
        self.thumbnails = []
        self.scroll_canvas = tk.Canvas(self, bg='#23272f', highlightthickness=0)
        self.scroll_canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar = tk.Scrollbar(self, orient='vertical', command=self.scroll_canvas.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner_frame = tk.Frame(self.scroll_canvas, bg='#23272f')
        self.scroll_canvas.create_window((0,0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind('<Configure>', lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox('all')))
        self.load_thumbnails()

    def load_thumbnails(self):
        files = sorted(glob.glob(self.image_pattern))
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.thumbnails = []
        cols = 4
        for idx, f in enumerate(files):
            img = Image.open(f)
            img.thumbnail((160, 120))
            imgtk = ImageTk.PhotoImage(img)
            lbl = tk.Label(self.inner_frame, image=imgtk, bg='#23272f', cursor='hand2')
            lbl.image = imgtk
            lbl.grid(row=idx//cols, column=idx%cols, padx=10, pady=10)
            lbl.bind('<Button-1>', lambda e, path=f: self.open_full_image(path))
            self.thumbnails.append(lbl)

    def open_full_image(self, path):
        win = tk.Toplevel(self)
        win.title(path)
        img = Image.open(path)
        imgtk = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=imgtk)
        lbl.image = imgtk
        lbl.pack()

class IntensityGraphPopup(tk.Toplevel):
    def __init__(self, parent, get_rms_func):
        super().__init__(parent)
        self.title('Audio Intensity Over Time')
        self.configure(bg='#23272f')
        self.geometry('600x350')
        self.get_rms_func = get_rms_func
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.closed = False
        self.fig, self.ax = plt.subplots(figsize=(7,3), facecolor='#23272f')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.rms_history = []
        self.max_history = 100
        self.after(200, self.update_plot)
    def update_plot(self):
        if self.closed:
            return
        rms = self.get_rms_func()
        self.rms_history.append(rms)
        if len(self.rms_history) > self.max_history:
            self.rms_history.pop(0)
        self.ax.clear()
        self.ax.plot(self.rms_history, color='#00e676')
        self.ax.set_xlabel('Time (blocks)', color='#fff')
        self.ax.set_ylabel('RMS Intensity', color='#fff')
        self.ax.set_title('Audio Intensity Over Time', color='#fff')
        self.ax.set_facecolor('#23272f')
        self.ax.tick_params(axis='x', colors='#fff')
        self.ax.tick_params(axis='y', colors='#fff')
        self.ax.spines['bottom'].set_color('#fff')
        self.ax.spines['top'].set_color('#fff')
        self.ax.spines['right'].set_color('#fff')
        self.ax.spines['left'].set_color('#fff')
        self.ax.set_ylim(0, max(0.1, max(self.rms_history)*1.1 if self.rms_history else 0.1))
        self.canvas.draw()
        self.after(200, self.update_plot)
    def close(self):
        self.closed = True
        self.destroy()

class AcousticCameraUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Camera 2D")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.latest_audio_block = np.zeros((BLOCK_SIZE, CHANNELS_TO_USE))
        self.bar_graph_popup = None
        self.setup_ui()
        self.is_running = True
        self.stream = None
        self.cap = None
        self.dark_mode = False
        self.apply_theme()
        self.min_freq_var.set(str(FREQUENCY_RANGE[0]))
        self.max_freq_var.set(str(FREQUENCY_RANGE[1]))
        self.start_processing()

    def setup_ui(self):
        self.main_frame = Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        # Left panel
        self.left_panel = Frame(self.main_frame, width=200)
        self.left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_label = Label(self.left_panel, text="Intensidade e Gráficos")
        left_label.pack(pady=(0, 10))
        self.intensity_btn = Button(self.left_panel, text="Mostrar Intensidade", command=self.show_intensity)
        self.intensity_btn.pack(fill="x", pady=5)
        self.graph_btn = Button(self.left_panel, text="Mostrar Gráfico", command=self.show_graph)
        self.graph_btn.pack(fill="x", pady=5)
        # Center panel (centered camera and controls)
        self.center_panel = Frame(self.main_frame)
        self.center_panel.pack(side="left", fill="both", expand=True)
        self.cam_label = Label(self.center_panel, text="CAM + BEAMFORMING")
        self.cam_label.pack(pady=(0, 10))
        self.canvas = tk.Canvas(self.center_panel, bg="black")
        self.canvas.pack(anchor='center', pady=10)
        # Frequency controls centered below camera
        self.freq_frame = Frame(self.center_panel)
        self.freq_frame.pack(anchor='center', pady=10)
        freq_label = Label(self.freq_frame, text="Frequency Range:")
        freq_label.pack(side="left", padx=(0, 10))
        min_label = Label(self.freq_frame, text="Mínimo(High-pass):")
        min_label.pack(side="left", padx=(0, 5))
        self.min_freq_var = tk.StringVar()
        self.min_freq_entry = Entry(self.freq_frame, textvariable=self.min_freq_var, width=8)
        self.min_freq_entry.pack(side="left", padx=(0, 10))
        max_label = Label(self.freq_frame, text="Máximo(Low-pass):")
        max_label.pack(side="left", padx=(0, 5))
        self.max_freq_var = tk.StringVar()
        self.max_freq_entry = Entry(self.freq_frame, textvariable=self.max_freq_var, width=8)
        self.max_freq_entry.pack(side="left")
        self.apply_btn = Button(self.freq_frame, text="Aplicar", command=self.apply_frequency)
        self.apply_btn.pack(side="left", padx=10)
        self.frames_btn = Button(self.center_panel, text="Frames", width=20, command=self.show_frames)
        self.frames_btn.pack(pady=10)
        # Right panel (remove Reiniciar Câmera)
        self.right_panel = Frame(self.main_frame, width=200)
        self.right_panel.pack(side="right", fill="y", padx=(10, 0))
        right_label = Label(self.right_panel, text="Funções")
        right_label.pack(pady=(0, 10))
        # Only keep the other buttons
        self.save_btn = Button(self.right_panel, text="Salvar Frame", command=self.save_image)
        self.save_btn.pack(fill="x", pady=5)
        self.record_btn = Button(self.right_panel, text="Gravar Vídeo", command=self.toggle_recording)
        self.record_btn.pack(fill="x", pady=5)
        self.theme_btn = Button(self.right_panel, text="Alternar Tema", command=self.toggle_theme)
        self.theme_btn.pack(fill="x", pady=5)
        self.flip_x_btn = Button(self.right_panel, text="Flip X", command=self.flip_x)
        self.flip_x_btn.pack(fill="x", pady=5)
        self.exit_btn = Button(self.right_panel, text="Sair", command=self.on_closing)
        self.exit_btn.pack(fill="x", pady=5)
        self.status_bar = Label(self.root, text="Pronto", bd=1, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")
        self.is_recording = False
        self.video_writer = None

    def apply_theme(self):
        # Define colors based on mode
        if self.dark_mode:
            bg_color = "#2E2E2E"
            fg_color = "#FFFFFF"
            button_bg = "#444444"
            canvas_bg = "#1E1E1E"
        else:
            bg_color = "#F0F0F0"
            fg_color = "#000000"
            button_bg = "#E0E0E0"
            canvas_bg = "#FFFFFF"
        
        # Apply to all widgets
        self.root.configure(bg=bg_color)
        self.main_frame.configure(bg=bg_color)
        self.left_panel.configure(bg=bg_color)
        self.center_panel.configure(bg=bg_color)
        self.right_panel.configure(bg=bg_color)
        self.freq_frame.configure(bg=bg_color)
        self.canvas.configure(bg=canvas_bg)
        self.status_bar.configure(bg=bg_color, fg=fg_color)
        
        # Apply to all labels
        for widget in self.root.winfo_children():
            self.update_widget_colors(widget, bg_color, fg_color, button_bg)
    
    def update_widget_colors(self, widget, bg_color, fg_color, button_bg):
        try:
            widget_type = widget.winfo_class()
            
            if widget_type in ('Label', 'Frame'):
                # Alguns widgets como Frame não suportam a opção fg
                if widget_type == 'Frame':
                    widget.configure(bg=bg_color)
                else:
                    widget.configure(bg=bg_color, fg=fg_color)
            elif widget_type == 'Button':
                widget.configure(bg=button_bg, fg=fg_color, activebackground=bg_color)
            elif widget_type == 'Entry':
                widget.configure(bg=button_bg, fg=fg_color, insertbackground=fg_color)
            elif widget_type == 'Canvas':
                widget.configure(bg=bg_color)
            
            # Recursivamente atualiza widgets filhos
            for child in widget.winfo_children():
                self.update_widget_colors(child, bg_color, fg_color, button_bg)
        except Exception as e:
            print(f"Erro ao configurar widget {widget.winfo_class()}: {e}")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        theme_name = "Escuro" if self.dark_mode else "Claro"
        self.status_bar.config(text=f"Tema {theme_name} aplicado")
    
    def start_processing(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        if not self.cap.isOpened():
            self.status_bar.config(text=f"Erro: Não foi possível abrir a câmera com ID {CAMERA_ID}")
            return
        device_id = find_device_id()
        if device_id is None:
            self.status_bar.config(text=f"Erro: Não foi possível encontrar o dispositivo de áudio '{DEVICE_NAME}'")
            return
        global bandpass_filter
        bandpass_filter = create_bandpass_filter(FREQUENCY_RANGE[0], FREQUENCY_RANGE[1], SAMPLE_RATE)
        def audio_callback_with_block(indata, frames, time, status):
            self.latest_audio_block = indata.copy()
            audio_callback(indata, frames, time, status)
        self.stream = sd.InputStream(
            device=device_id, 
            channels=CHANNELS_TO_USE, 
            samplerate=SAMPLE_RATE, 
            blocksize=BLOCK_SIZE, 
            callback=audio_callback_with_block, 
            latency='low'
        )
        self.stream.start()
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        self.status_bar.config(text="Câmera acústica 2D iniciada. Processando...")
    
    def process_video(self):
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            
            # Create a transparent overlay layer
            overlay = frame.copy()
            final_frame = frame.copy()
            
            if latest_power_map is not None:
                # Find the peak power and its location
                peak_v_idx, peak_h_idx = np.unravel_index(np.argmax(latest_power_map), latest_power_map.shape)
                peak_power = latest_power_map[peak_v_idx, peak_h_idx]
                
                h_scan_range = np.arange(*H_SCAN_ANGLES)
                v_scan_range = np.arange(*V_SCAN_ANGLES)
                
                peak_h_angle = h_scan_range[peak_h_idx]
                peak_v_angle = v_scan_range[peak_v_idx]
                
                # Draw heatmap overlay
                heatmap = cv2.resize((latest_power_map * 255).astype(np.uint8), 
                                    (frame_width, frame_height), 
                                    interpolation=cv2.INTER_CUBIC)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                final_frame = cv2.addWeighted(heatmap_color, HOTSPOT_ALPHA, frame, 1 - HOTSPOT_ALPHA, 0)
                
                # Display peak angle text
                info_text = f"PEAK H: {peak_h_angle:.1f}  V: {peak_v_angle:.1f}"
                cv2.putText(final_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # Convert to RGB for tkinter
            rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.canvas.config(width=frame_width, height=frame_height)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.image = imgtk  # Keep a reference
            
            # Record video if enabled
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(final_frame)
            
            time.sleep(0.01)  # Small delay to reduce CPU usage
    
    def apply_frequency(self):
        try:
            min_freq = int(self.min_freq_var.get())
            max_freq = int(self.max_freq_var.get())
            
            if min_freq >= max_freq:
                self.status_bar.config(text="Erro: Frequência mínima deve ser menor que a máxima")
                return
            
            global FREQUENCY_RANGE, bandpass_filter
            FREQUENCY_RANGE = [min_freq, max_freq]
            bandpass_filter = create_bandpass_filter(min_freq, max_freq, SAMPLE_RATE)
            
            self.status_bar.config(text=f"Filtro de frequência aplicado: {min_freq}Hz - {max_freq}Hz")
        except ValueError:
            self.status_bar.config(text="Erro: Valores de frequência inválidos")
    
    def show_intensity(self):
        if not hasattr(self, 'intensity_popup') or not self.intensity_popup or not self.intensity_popup.winfo_exists():
            self.intensity_popup = IntensityGraphPopup(self.root, self.get_latest_rms)
            self.intensity_popup.grab_set()
        else:
            self.intensity_popup.lift()

    def get_latest_rms(self):
        # Return RMS of all microphones combined for the latest audio block
        if self.latest_audio_block is not None:
            return float(np.sqrt(np.mean(self.latest_audio_block**2)))
        return 0.0
    
    def show_graph(self):
        if self.bar_graph_popup is None or not self.bar_graph_popup.winfo_exists():
            self.bar_graph_popup = MicBarGraphPopup(self.root, self.get_latest_mic_intensities)
            self.bar_graph_popup.grab_set()  # Modal
        else:
            self.bar_graph_popup.lift()

    def get_latest_mic_intensities(self):
        # Return RMS for each channel in the latest audio block
        return np.sqrt(np.mean(self.latest_audio_block**2, axis=0)) if self.latest_audio_block is not None else np.zeros(CHANNELS_TO_USE)
    
    def show_frames(self):
        FramesPopup(self.root)
    
    def restart_camera(self):
        # Reiniciar a câmera
        if self.cap is not None:
            self.cap.release()
            self.cap = cv2.VideoCapture(CAMERA_ID)
            self.status_bar.config(text="Câmera reiniciada")
    
    def save_image(self):
        # Salvar a imagem atual
        if self.cap is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"acoustic_image_{timestamp}.png"
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(filename, frame)
                self.status_bar.config(text=f"Imagem salva como {filename}")
    
    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"acoustic_video_{timestamp}.avi"
            
            if self.cap is not None:
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = 20.0
                
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
                
                self.is_recording = True
                self.record_btn.config(text="Parar Gravação")
                self.status_bar.config(text=f"Gravando vídeo para {filename}")
        else:
            # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            self.is_recording = False
            self.record_btn.config(text="Gravar Vídeo")
            self.status_bar.config(text="Gravação de vídeo finalizada")
    
    def flip_x(self):
        global mic_positions
        mic_positions = mic_positions.copy()
        mic_positions[:, 0] *= -1
        print("Flipped X:", mic_positions)
        self.status_bar.config(text="Eixo X invertido (Flip X)")
    
    def on_resize(self, event):
        # Resize the canvas to fill the center panel
        if self.center_panel.winfo_width() > 0 and self.center_panel.winfo_height() > 0:
            self.canvas.config(width=self.center_panel.winfo_width(), height=self.center_panel.winfo_height())
    
    def on_closing(self):
        self.is_running = False
        
        # Clean up resources
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        
        if self.cap is not None:
            self.cap.release()
        
        if self.video_writer is not None:
            self.video_writer.release()
        
        self.root.destroy()
        print("Aplicação encerrada.")

# --- Main Program Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AcousticCameraUI(root)
    root.mainloop()
    
    # Clean up OpenCV windows if any are left open
    cv2.destroyAllWindows()
    print("Aplicação encerrada.")
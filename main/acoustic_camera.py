import numpy as np
import sounddevice as sd
import cv2
from scipy.signal import butter, lfilter

# --- CONFIGURATION ---
SAMPLE_RATE = 48000  # UMA-16 default
BLOCK_SIZE = 1024    # Audio block size
N_MICS = 16          # UMA-16
CAM_INDEX = 0        # Default webcam
AUDIO_DEVICE = None  # Set to None for default, or use sd.query_devices() to find UMA-16 index
MIC_SPACING = 0.01   # 1 cm spacing (meters)
SOUND_SPEED = 343.0  # Speed of sound in air (m/s)

# Beamforming parameters
N_ANGLES = 90  # Number of look directions (for heatmap)
angles = np.linspace(-np.pi/2, np.pi/2, N_ANGLES)  # -90 to +90 degrees

# UMA-16 mic positions (linear array along x)
mic_positions = np.zeros((N_MICS, 3))
mic_positions[:, 0] = np.arange(N_MICS) * MIC_SPACING

# --- High-pass filter ---
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data, axis=-1)

# --- AUDIO CALLBACK ---
audio_buffer = np.zeros((N_MICS, BLOCK_SIZE))
def audio_callback(indata, frames, time, status):
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

# --- MAIN LOOP ---
def main():
    # Open webcam
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=N_MICS,
        device=AUDIO_DEVICE,
        callback=audio_callback
    )
    stream.start()

    print("Press 'q' in the video window to quit.")

    # High-pass filter parameters
    HPF_CUTOFF = 500  # Lowered to 500 Hz for more sensitivity
    HPF_ORDER = 4

    # Smoothing buffers
    smooth_intensity = 0.0
    smooth_idx = 0.0
    alpha = 0.5  # Smoothing factor (0=slow, 1=fast)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Copy and high-pass filter audio buffer
        audio = audio_buffer.copy()
        audio = highpass_filter(audio, HPF_CUTOFF, SAMPLE_RATE, HPF_ORDER)

        # --- Beamforming ---
        intensities = delay_and_sum(audio, mic_positions, angles, SAMPLE_RATE, SOUND_SPEED)
        abs_max_intensity = np.max(intensities)
        max_idx = np.argmax(intensities)

        # Smoothing
        smooth_intensity = alpha * abs_max_intensity + (1 - alpha) * smooth_intensity
        smooth_idx = alpha * max_idx + (1 - alpha) * smooth_idx

        abs_threshold = 0.01  # Lowered for more sensitivity
        overlay = frame.copy()  # Always define overlay
        print(f"smooth_intensity: {smooth_intensity:.4f}")  # Debug print
        if smooth_intensity > abs_threshold:
            # Color: blue (low) to red (high)
            color_map = cv2.applyColorMap(np.array([int(np.clip(smooth_intensity * 255, 0, 255))], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
            x_pos = int((smooth_idx / (len(angles) - 1)) * (frame.shape[1] - 1))
            y_pos = frame.shape[0] // 2
            radius = int(30 + 30 * np.clip(smooth_intensity, 0, 1))
            cv2.circle(overlay, (x_pos, y_pos), radius, color_map, -1)
            spot_alpha = 0.5 * np.clip(smooth_intensity, 0, 1) + 0.2
            overlay = cv2.addWeighted(overlay, spot_alpha, frame, 1 - spot_alpha, 0)
        cv2.imshow('Acoustic Camera', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
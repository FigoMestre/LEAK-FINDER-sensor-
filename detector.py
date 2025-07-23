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
    [-0.063,  0.075], [-0.021,  0.075], [0.021,  0.075], [0.063,  0.075],
    [-0.063,  0.025], [-0.021,  0.025], [0.021,  0.025], [0.063,  0.025],
    [-0.063, -0.025], [-0.021, -0.025], [0.021, -0.025], [0.063, -0.025],
    [-0.063, -0.075], [-0.021, -0.075], [0.021, -0.075], [0.063, -0.075]
])

# --- MIRRORING OPTIONS FOR TROUBLESHOOTING ---
# If the beamforming is mirrored horizontally, uncomment the next line:
# mic_positions[:, 0] *= -1
# If the beamforming is mirrored vertically, uncomment the next line:
# mic_positions[:, 1] *= -1

# Processing & Visualization Settings
# Coarse resolution for real-time performance. Decrease step for more precision.
H_SCAN_ANGLES = (-70, 71, 15)  # Scan horizontally in 15-degree steps
V_SCAN_ANGLES = (-45, 46, 15)  # Scan vertically in 15-degree steps
SPEED_OF_SOUND = 343.0         # m/s
HOTSPOT_ALPHA = 0.5            # Transparency of the hotspot circle

# --- Global variable to share data between threads ---
latest_power_map = None
smoothed_power_map = None
SMOOTHING_FACTOR = 0.8  # Between 0 (no smoothing) and 1 (very smooth)

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
    """This function is called by sounddevice in a separate thread."""
    global latest_power_map
    if status:
        print(f"Audio Status: {status}")

    filtered_audio = sosfilt(bandpass_filter, indata, axis=0)

    h_angles = np.deg2rad(np.arange(*H_SCAN_ANGLES))
    v_angles = np.deg2rad(np.arange(*V_SCAN_ANGLES))
    power_map = np.zeros((len(v_angles), len(h_angles)))

    # --- Nested loop for 2D scanning ---
    for v_idx, v_angle in enumerate(v_angles):
        for h_idx, h_angle in enumerate(h_angles):
            # Calculate a direction vector for the current scan angle
            dir_vector = np.array([np.cos(v_angle) * np.sin(h_angle), np.sin(v_angle)])

            # Calculate time delays for each mic using dot product
            time_delays = np.dot(mic_positions, dir_vector) / SPEED_OF_SOUND
            sample_shifts = np.round(time_delays * SAMPLE_RATE).astype(int)

            summed_signal = np.zeros(BLOCK_SIZE, dtype=np.float32)
            for ch in range(CHANNELS_TO_USE):
                summed_signal += np.roll(filtered_audio[:, ch], -sample_shifts[ch])

            power_map[v_idx, h_idx] = np.sum(summed_signal**2)

    # Remove background offset
    power_map = power_map - np.min(power_map)

    # Normalize
    if np.max(power_map) > 0:
        power_map = power_map / np.max(power_map)

    # Apply a threshold to suppress noise
    threshold = 0.2  # Try values between 0.1 and 0.3
    power_map[power_map < threshold] = 0

    global smoothed_power_map
    if smoothed_power_map is None:
        smoothed_power_map = power_map
    else:
        smoothed_power_map = SMOOTHING_FACTOR * smoothed_power_map + (1 - SMOOTHING_FACTOR) * power_map

    latest_power_map = smoothed_power_map

# --- Main Program Execution ---
if __name__ == "__main__":
    device_id = find_device_id()
    if device_id is None:
        print(f"❌ Error: Could not find audio device matching '{DEVICE_NAME}'. Exiting.")
        exit()

    bandpass_filter = create_bandpass_filter(FREQUENCY_RANGE[0], FREQUENCY_RANGE[1], SAMPLE_RATE)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera with ID {CAMERA_ID}. Exiting.")
        exit()

    print("\n🚀 Starting 2D Acoustic Camera. Press 'q' to quit.")

    with sd.InputStream(device=device_id, channels=CHANNELS_TO_USE, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, callback=audio_callback, latency='low'):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape

            cv2.namedWindow('2D Acoustic Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('2D Acoustic Camera', frame_width, frame_height)

            # Create a transparent overlay layer to draw on
            overlay = frame.copy()
            final_frame = frame.copy()

            if latest_power_map is not None:
                # 1. Find the peak power and its location in the 2D map
                peak_v_idx, peak_h_idx = np.unravel_index(np.argmax(latest_power_map), latest_power_map.shape)
                peak_power = latest_power_map[peak_v_idx, peak_h_idx]

                h_scan_range = np.arange(*H_SCAN_ANGLES)
                v_scan_range = np.arange(*V_SCAN_ANGLES)

                peak_h_angle = h_scan_range[peak_h_idx]
                peak_v_angle = v_scan_range[peak_v_idx]

                # --- Draw heatmap overlay instead of hotspot ball ---
                # Resize power_map to match frame size
                heatmap = cv2.resize((latest_power_map * 255).astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # Blend the heatmap with the frame
                final_frame = cv2.addWeighted(heatmap_color, HOTSPOT_ALPHA, frame, 1 - HOTSPOT_ALPHA, 0)

                # 3. Display the peak angle text regardless of power
                info_text = f"PEAK H: {peak_h_angle:.1f}  V: {peak_v_angle:.1f}"
                cv2.putText(final_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Show the final image
            cv2.imshow('2D Acoustic Camera', final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Application stopped.")
import sounddevice as sd
import numpy as np
import time

# --- Configuration ---
DEVICE_NAME = 'UMA16'  # Change if your device has a different name
SAMPLE_RATE = 48000
CHANNELS = 16
DURATION = 2  # seconds to record per mic

# Custom arrangement for reference
arrangement = [
    [8, 7, 10, 9],
    [6, 5, 12, 11],
    [4, 3, 14, 13],
    [2, 1, 16, 15]
]

print("\nMicrophone arrangement (by channel number):")
for row in arrangement:
    print('  '.join(f"{n:2}" for n in row))
print("\nEach channel will be tested in order (1-16). Listen for the playback to verify each mic.")

# --- List all input devices ---
print("\nAvailable audio input devices:")
devices = sd.query_devices()
input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] >= CHANNELS]
for i, d in input_devices:
    print(f"ID {i}: {d['name']} (inputs: {d['max_input_channels']})")

# --- Select device ---
if not input_devices:
    print("No suitable input devices found with at least 16 channels.")
    exit(1)

# Try to auto-select by name, else ask user
selected_id = None
for i, d in input_devices:
    if DEVICE_NAME.lower() in d['name'].lower():
        selected_id = i
        print(f"\nAuto-selected device: {d['name']} (ID {i})")
        break
if selected_id is None:
    selected_id = int(input("\nEnter the device ID to use: "))

print(f"\nRecording {DURATION} seconds from all {CHANNELS} channels...")
recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, device=selected_id)
sd.wait()
print("Recording complete.\n")

# --- Analyze and save results ---
results = []
for ch in range(CHANNELS):
    mic_num = ch + 1
    data = recording[:, ch]
    # Volume (RMS)
    rms = np.sqrt(np.mean(data**2))
    # Dominant frequency
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1/SAMPLE_RATE)
    idx = np.argmax(np.abs(fft))
    dom_freq = freqs[idx]
    results.append((mic_num, rms, dom_freq))

# Save to TXT file
with open('mic_test_results.txt', 'w') as f:
    f.write('Mic\tVolume(RMS)\tDominant Frequency (Hz)\n')
    for mic_num, rms, dom_freq in results:
        f.write(f'{mic_num}\t{rms:.6f}\t{dom_freq:.1f}\n')

print("Results saved to mic_test_results.txt.\n")

# --- Test each channel (playback, optional) ---
for ch in range(CHANNELS):
    mic_num = ch + 1
    print(f"Testing Mic {mic_num} (Channel {ch+1}) - listen now...")
    sd.play(recording[:, ch], samplerate=SAMPLE_RATE)
    sd.wait()
    time.sleep(0.5)
    print(f"Mic {mic_num} test done.\n")

print("All microphones tested. Refer to the arrangement above to identify physical positions.")
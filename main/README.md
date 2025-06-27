# Acoustic Camera (Low Cost)

This project is a real-time acoustic camera using a USB webcam and a miniDSP UMA-16 (16-mic array) on Windows 11. It uses Python, Acoular (for beamforming), and OpenCV for visualization.

## Hardware Requirements
- miniDSP UMA-16 (16-channel USB microphone array)
- USB webcam
- Windows 11 (should work on other OSes with minor changes)

## Software Requirements
- Python 3.10+
- Conda (recommended for environment management)

## Setup Instructions

1. **Clone this repository** (or copy the files to a folder):
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda create -n acoustic_camera python=3.10 numpy scipy opencv matplotlib
   conda activate acoustic_camera
   ```

3. **Install Acoular (via pip):**
   ```bash
   pip install acoular
   pip install sounddevice
   pip install pyyaml

   ```
   If you have issues, see the [Acoular install guide](https://acoular.org/install.html).

4. **Run the application:**
   ```bash
   set OPENBLAS_NUM_THREADS=1
   python acoustic_camera.py
   ```

## Usage
- The app will open a window showing the webcam feed with a sound intensity heatmap overlay.
- Speak, clap, or make noise near the UMA-16 to see the sound source visualized.
- Press `q` in the window to quit.

## Troubleshooting
- Ensure your UMA-16 is set as the default audio input device, or modify the script to select the correct device index.
- If you get errors about missing channels, check your audio device settings.
- Acoular may require additional dependencies (e.g., h5py, PyTables). Install them via pip if prompted.

## Credits
- [Acoular](https://acoular.org/) for beamforming
- [OpenCV](https://opencv.org/) for video and visualization 
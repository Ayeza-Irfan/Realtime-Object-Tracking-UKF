# Real-Time Object Tracking using Unscented Kalman Filter (UKF)

A real-time object tracking system that combines **color-based detection (OpenCV)** with an **Unscented Kalman Filter (UKF)** for robust, smoothed position estimation under noisy or nonlinear motion. Tracks a colored object live via webcam and visualizes detected vs. estimated trajectories after the session ends.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-✓-green)
![FilterPy](https://img.shields.io/badge/FilterPy-UKF-orange)

## Features

- **Real-time color-based object detection** using HSV color thresholding and contour analysis to locate the largest matching blob in each frame.
- **Unscented Kalman Filter (UKF) state estimation** with a constant-velocity motion model, using `filterpy`'s `MerweScaledSigmaPoints` for sigma-point generation — handles nonlinear motion and noisy measurements better than a standard Kalman Filter.
- **Live visual overlay** distinguishing the raw detected position (red) from the UKF-smoothed estimate (green) directly on the video feed.
- **Post-session trajectory plotting** — once tracking ends, the full detected path and UKF-estimated path are plotted with Matplotlib for visual comparison.
- **Tunable noise parameters** (process noise `Q`, measurement noise `R`) to adjust filter responsiveness vs. smoothness.

## How It Works

1. Each webcam frame is converted to HSV color space, and a mask isolates pixels within a defined blue color range.
2. Contours are extracted from the mask, and the largest contour above a minimum area threshold is treated as the tracked object.
3. The object's centroid (via image moments) is passed to the UKF as a noisy position measurement.
4. The UKF performs a **predict** step (propagating state via the constant-velocity model) followed by an **update** step (correcting with the new measurement), producing a smoothed position estimate.
5. Both the raw detection and the UKF estimate are drawn on the live feed, and their full trajectories are stored for plotting after the session ends.

## Tech Stack

| Component             | Technology         |
|------------------------|---------------------|
| Object Detection        | OpenCV (HSV thresholding, contours) |
| State Estimation        | FilterPy (Unscented Kalman Filter)   |
| Visualization            | Matplotlib            |
| Core Language            | Python, NumPy          |

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ayeza-Irfan/Realtime-Object-Tracking-UKF
.git
  
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python numpy filterpy matplotlib
   ```

## Usage

Run the tracker:

```bash
python main.py
```

1. A webcam window opens, tracking any object matching the predefined blue color range.
2. The **red dot** marks the raw detected centroid; the **green circle** marks the UKF's smoothed estimate.
3. Press **`ESC`** to stop tracking.
4. After exiting, a Matplotlib plot opens comparing the raw detected path against the UKF-estimated path.

## Customization

- **Tracking a different color**: adjust the `lower` and `upper` HSV bounds in the script to match your target object's color.
- **Filter tuning**: modify `ukf.Q` (process noise) and `ukf.R` (measurement noise) to make the filter more responsive (lower noise values) or more stable/smoothed (higher noise values).
- **Motion model**: the current model (`fx`) assumes constant velocity; this can be extended to constant acceleration or other nonlinear models for more complex motion patterns.

## Notes & Limitations

- Requires a working webcam accessible via OpenCV (`cv2.VideoCapture(0)`).
- Detection relies on a fixed HSV color range, so tracking accuracy depends on lighting conditions and the chosen object's color consistency.
- Designed for single-object tracking; multiple matching-color objects will only track the largest contour.

## Author

Built by **Ayeza Irfan**.

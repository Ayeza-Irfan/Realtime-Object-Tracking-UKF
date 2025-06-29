import cv2
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt

# Define motion model (constant velocity)
def fx(x, dt):
    return np.array([
        x[0] + x[2] * dt,
        x[1] + x[3] * dt,
        x[2],
        x[3]
    ])

# Measurement model: we observe only position
def hx(x):
    return x[:2]

# UKF initialization
dt = 1.0
points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=0)
ukf = UKF(dim_x=4, dim_z=2, fx=fx, hx=hx, dt=dt, points=points)
ukf.x = np.array([0., 0., 0., 0.])
ukf.P *= 100
ukf.R = np.eye(2) * 25  # measurement noise
ukf.Q = np.eye(4) * 0.1  # process noise

# Store path for plotting
true_path = []
ukf_path = []

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Webcam not detected!")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue color mask
        lower = np.array([100, 150, 50])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    measurement = np.array([cx, cy])

                    # UKF processing
                    ukf.predict()
                    ukf.update(measurement)

                    # Save positions
                    true_path.append((cx, cy))
                    ukf_path.append((ukf.x[0], ukf.x[1]))

                    # Draw tracking
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)        # Red: Detected center
                    cv2.circle(frame, (int(ukf.x[0]), int(ukf.x[1])), 8, (0, 255, 0), 2)  # Green: UKF estimate

        cv2.imshow("UKF Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break  # ESC to exit

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Plot results after exit
    if true_path and ukf_path:
        true_path = np.array(true_path)
        ukf_path = np.array(ukf_path)

        plt.figure(figsize=(10, 6))
        plt.plot(true_path[:, 0], true_path[:, 1], 'r-', label='Detected Path')
        plt.plot(ukf_path[:, 0], ukf_path[:, 1], 'g--', label='UKF Estimate')
        plt.title('Real Object Tracking using UKF and OpenCV')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

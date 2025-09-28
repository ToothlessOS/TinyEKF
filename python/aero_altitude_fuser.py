import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tinyekf import EKF

# Convert ASL m to Pascals: see
# http://www.engineeringtoolbox.com/air-altitude-pressure-d_462.html
def asl2baro(asl):
    return 101325 * pow((1 - 2.25577e-5 * asl), 5.25588)

# Convert Pascals to m ASL
def baro2asl(pa):
    return (1.0 - pow(pa / 101325.0, 0.190295)) * 44330.0

# Read flight data from CSV
df = pd.read_csv(r'data\2025-06-13-serial-6769-flight-0006-via-16608(in).csv')
timestamps = df['time'].to_numpy()
delta_t = np.diff(timestamps, prepend=timestamps[0])
accel = df['acceleration'].to_numpy()
baro = df['pressure']
altitude = baro.apply(baro2asl).to_numpy()
ref_altitude = df['altitude'].to_numpy() # From flight computer; For validation!
speed = df['speed'].to_numpy()

# Run EKF
N = len(timestamps)

## Simple EKF (version 0)
## State: altitude (m ASL), velocity (m/s)
## Measurements: barometric altitude (m ASL)
P = np.eye(2) * 1e-1 # Tunable params
Q = np.eye(2) * 1e-4
R = np.eye(1) * 5e-1

fused = np.zeros(N)

ekf = EKF(P)
ekf.x = np.array([altitude[0], 0]) # Initial state: altitude from baro, zero velocity


for k in range(N):
    z = altitude[k]
    # state transition function
    dt = delta_t[k]
    # x[0]: altitude, x[1]: velocity
    fx = np.array([ekf.x[0] + ekf.x[1] * dt + 0.5 * accel[k] * dt * dt,
                   ekf.x[1] + accel[k] * dt])
    # Jacobian of the state transition function
    F = np.array([[1, dt],
                  [0, 1]])
    
    # measurement function
    hx = np.array([ekf.x[0]])
    # Jacobian of the measurement function
    H = np.array([[1, 0]])

    # Run EKF steps
    ekf.predict(fx, F, Q)

    ekf.update(z, hx, H, R)

    # Collect fused altitude
    fused[k] = ekf.get()[0]

# Plot altitude and speed in side-by-side subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Altitude subplot
axs[0].plot(timestamps, altitude, label='Barometric Altitude (m ASL)', color='red')
axs[0].plot(timestamps, fused, label='Fused Altitude (m ASL)', color='blue')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Altitude (m ASL)')
axs[0].legend()
axs[0].set_title('Altitude')

# Speed subplot
axs[1].plot(timestamps, speed, label='Speed (m/s)', color='green')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Speed (m/s)')
axs[1].legend()
axs[1].set_title('Speed')

plt.tight_layout()
plt.show()

"""
============================================================
 SmartGlove — 7-Stage Early Intention Prediction Pipeline
 Hardware: Pi 5 + BNO055 + Logitech C270 (on glove) + ArUco board
 Configuration: Camera ON glove, ArUco markers FIXED in environment
============================================================

STAGE MAPPING (from BTP doc):
  Stage 1 — Physical Interaction Modelling (object positions from ArUco)
  Stage 2 — Sensor Data Acquisition      (BNO055 @ 100Hz + Camera @ 30Hz)
  Stage 3 — Motion Onset Detection       (EKF velocity > 0.02 m/s)
  Stage 4 — State Estimation via EKF     (6-state: pos + vel, BNO055 orientation)
  Stage 5 — Early Motion Feature Extract (θ₁,θ₂,d₁,d₂,ḋ₁,ḋ₂,‖v‖)
  Stage 6 — Probabilistic Goal Inference (softmax scoring)
  Stage 7 — MuJoCo validation (simulation only — not needed here)

CAMERA-ON-GLOVE LOGIC:
  - Camera moves with the hand → hand position = camera position (from solvePnP)
  - ArUco markers are FIXED in world → their world positions are known constants
  - K1 and K2 object positions are derived from marker geometry (fixed in world)
  - EKF tracks HAND position in world frame (= camera position from inverse pose)
  - BNO055 gives absolute orientation → rotates linear_accel to world frame
  - EKF predict: BNO055 accel @ 100Hz
  - EKF correct: solvePnP hand position @ camera frame rate (~20-30Hz)
"""

import cv2
import asyncio
import websockets
import base64
import json
import time
import numpy as np
import os

import board
import busio
import adafruit_bno055

try:
    from gpiozero import LED
    led_green = LED(17)   # GPIO17 = Pin 11 (green LED → K1)
    led_red   = LED(27)   # GPIO27 = Pin 13 (red LED → K2)
    LEDS_OK = True
    print("[HW] LEDs initialised: GPIO17=green(K1), GPIO27=red(K2)")
except Exception as e:
    LEDS_OK = False
    print(f"[HW] LED init skipped: {e}")
    class _FakeLED:
        def on(self): pass
        def off(self): pass
    led_green = _FakeLED()
    led_red   = _FakeLED()

# ============================================================
# HARDWARE CONSTANTS
# ============================================================
MARKER_SIZE      = 0.10       # 10cm printed ArUco markers
MARKER_IDS       = [0, 1, 2, 3]  # IDs on board: 0=TL, 1=TR, 2=BL, 3=BR

# Board geometry (measure your actual board)
# Markers are 30cm apart horizontally and vertically (centre to centre)
BOARD_H_SPACING  = 0.30       # horizontal distance between left and right markers (m)
BOARD_V_SPACING  = 0.30       # vertical distance between top and bottom markers (m)

# K1 and K2 object offsets from board centre
# K1 is the LEFT object (between ID0 and ID2), K2 is the RIGHT (between ID1 and ID3)
# Objects sit 5cm IN FRONT of the board face (toward camera side)
K1_OFFSET_X = -BOARD_H_SPACING / 4   # left of board centre
K2_OFFSET_X = +BOARD_H_SPACING / 4   # right of board centre
OBJ_FORWARD  = 0.05                   # 5cm in front of board plane

# EKF parameters
DT_IMU       = 1.0 / 100             # BNO055 at 100 Hz
DT_CAM       = 1.0 / 25             # camera correction target ~25 Hz
SIGMA_POS    = 0.005                  # camera position noise: 5mm
SIGMA_VEL    = 0.02                  # velocity noise floor

# Onset + prediction
ONSET_THRESH = 0.02                  # m/s — motion onset threshold
PREDICT_AT   = 0.300                 # seconds after onset
COOLDOWN     = 3.0                   # seconds between predictions
HISTORY_SIZE = 10                    # 100ms lookback at 100Hz

# Scoring weights (from BTP doc)
W_THETA = -3.0
W_DDOT  = -0.5
W_DIST  = -0.3
TEMP    =  3.0

# Camera calibration (load if available, otherwise use approximate)
CAM_MTX  = None
DIST_COEFFS = None
_calib_path = os.path.expanduser("~/camera_matrix.npy")
_dist_path  = os.path.expanduser("~/dist_coeffs.npy")
if os.path.exists(_calib_path) and os.path.exists(_dist_path):
    CAM_MTX     = np.load(_calib_path)
    DIST_COEFFS = np.load(_dist_path)
    print(f"[CAL] Loaded calibration from ~/camera_matrix.npy")
else:
    # Approximate for C270 at 640×480
    fx = fy = 600.0
    cx, cy = 320.0, 240.0
    CAM_MTX     = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    DIST_COEFFS = np.zeros((5,), dtype=np.float64)
    print("[CAL] No calibration file — using approximate C270 matrix (run calibrate.py for accuracy)")

# ============================================================
# STAGE 4 — EXTENDED KALMAN FILTER (6-state, BNO055 version)
# ============================================================
# State: x = [px, py, pz,  vx, vy, vz]  (world frame)
# Orientation is handled by BNO055 onboard fusion — NOT in EKF state
# BNO055 provides absolute quaternion → rotate linear_accel to world frame
# Camera provides absolute hand position (from solvePnP inverse) for correction

class EKF6:
    def __init__(self, initial_pos: np.ndarray, dt: float):
        self.dt = dt
        self.x  = np.zeros(6)
        self.x[0:3] = initial_pos.copy()
        self.P  = np.eye(6) * 5
        self.Q  = np.diag([1e-5, 1e-5, 1e-5,   # position process noise
                           1e-3, 1e-3, 1e-3])   # velocity process noise
        # Camera corrects position only (3-dim)
        self.R_cam = np.eye(3) * (SIGMA_POS ** 2)
        self.H     = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)

    @staticmethod
    def quat_to_rot(q: np.ndarray) -> np.ndarray:
        """BNO055 quaternion [w, x, y, z] → 3×3 rotation matrix."""
        qw, qx, qy, qz = q
        return np.array([
            [1-2*(qy**2+qz**2),  2*(qx*qy-qw*qz),  2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),  2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy),    2*(qy*qz+qw*qx),  1-2*(qx**2+qy**2)]
        ], dtype=np.float64)

    def predict(self, a_linear_bno: np.ndarray, q_bno: np.ndarray):
        """
        Stage 4 prediction step — runs every BNO055 sample (100 Hz).
        a_linear_bno: gravity-compensated acceleration in BODY frame
        q_bno:        absolute orientation quaternion [w,x,y,z]
        """
        R       = self.quat_to_rot(q_bno)
        a_world = R @ a_linear_bno            # rotate to world frame

        # Integrate
        self.x[0:3] += self.x[3:6] * self.dt
        self.x[3:6] += a_world * self.dt

        # Covariance propagation with 6×6 Jacobian
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * self.dt
        self.P = F @ self.P @ F.T + self.Q

    def correct(self, z_pos: np.ndarray):
        """
        Stage 4 correction step — runs when camera gives hand position (~25 Hz).
        z_pos: [x, y, z] hand position in WORLD frame
        """
        y = z_pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R_cam
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x     = self.x + K @ y
        self.P     = (np.eye(6) - K @ self.H) @ self.P

    def get_pos(self) -> np.ndarray: return self.x[0:3].copy()
    def get_vel(self) -> np.ndarray: return self.x[3:6].copy()


# ============================================================
# STAGE 1 — PHYSICAL INTERACTION MODELLING
# ============================================================
# We need to establish the world-frame positions of K1 and K2.
# Since camera is ON the glove, the ArUco markers are fixed in the world.
# From multiple solvePnP solutions we average each marker's world position.
# Once we have all 4 markers stable, K1/K2 positions can be computed.

class BoardTracker:
    """
    Maintains stable world-frame positions of the 4 ArUco markers.
    On camera-on-glove: hand_pos_world = camera_pos_world (inverse of camera pose).
    Each frame: solvePnP per marker → camera pose in world → hand position.
    """

    def __init__(self):
        # 3D corners of a single marker (in marker's local frame, centred at marker)
        hs = MARKER_SIZE / 2.0
        self.marker_obj_pts = np.array([
            [-hs,  hs, 0],
            [ hs,  hs, 0],
            [ hs, -hs, 0],
            [-hs, -hs, 0]
        ], dtype=np.float64)

        # Known world positions of each marker (ID 0-3)
        # Board coordinate system: origin at board centre, Z forward (toward camera)
        # ID0=TL, ID1=TR, ID2=BL, ID3=BR
        hw = BOARD_H_SPACING / 2.0
        hv = BOARD_V_SPACING / 2.0
        self.marker_world_pos = {
            0: np.array([-hw,  hv, 0.0]),  # top-left
            1: np.array([ hw,  hv, 0.0]),  # top-right
            2: np.array([-hw, -hv, 0.0]),  # bottom-left
            3: np.array([ hw, -hv, 0.0]),  # bottom-right
        }

        # K1 and K2 object positions in world frame
        # K1: left object, K2: right object, both in front of board
        self.K1_world = np.array([K1_OFFSET_X, 0.0, OBJ_FORWARD])
        self.K2_world = np.array([K2_OFFSET_X, 0.0, OBJ_FORWARD])

        self.last_hand_pos = None
        self.board_visible  = False

    def process_frame(self, corners, ids):
        """
        Given detected corners and IDs, compute:
          1. hand position in world frame (camera-on-glove = camera pos in world)
          2. 3D marker info for dashboard

        Returns:
          hand_pos_world: np.ndarray(3,) or None
          markers_info:   list of dicts with id, cx, cy, tz, world_pos
        """
        markers_info = []
        hand_positions = []  # collect hand pos from each visible marker, then average

        if ids is None:
            self.board_visible = False
            return None, markers_info

        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid not in self.marker_world_pos:
                continue

            # Image corners for this marker
            img_pts = corners[i][0].astype(np.float64)
            cx = int(img_pts[:, 0].mean())
            cy = int(img_pts[:, 1].mean())

            # solvePnP: marker in camera frame
            ret, rvec, tvec = cv2.solvePnP(
                self.marker_obj_pts, img_pts,
                CAM_MTX, DIST_COEFFS,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ret:
                continue

            # Camera position in MARKER frame
            # tvec = position of marker origin in camera frame
            # To get camera position in world frame:
            #   R_cam_in_marker = R_marker_in_cam.T
            #   t_cam_in_marker = -R_cam_in_marker @ tvec
            # Then transform to world using known marker world position

            R_marker_in_cam, _ = cv2.Rodrigues(rvec)
            # Camera position in marker frame
            cam_pos_in_marker = -R_marker_in_cam.T @ tvec.flatten()
            # Transform to world frame
            # World frame: marker's world rotation assumed to be identity
            # (board is upright, world Z is toward camera)
            # So: hand_pos_world = marker_world_pos + cam_pos_in_marker
            marker_wpos = self.marker_world_pos[mid]
            hand_pos_world = marker_wpos + cam_pos_in_marker

            hand_positions.append(hand_pos_world)
            tz_cm = abs(tvec[2][0]) * 100.0  # distance from camera to marker in cm

            markers_info.append({
                "id":    mid,
                "cx":    cx,
                "cy":    cy,
                "tz_cm": round(tz_cm, 1),
                "world": marker_wpos.tolist()
            })

        if hand_positions:
            # Average hand position from all visible markers (more stable)
            self.last_hand_pos = np.mean(hand_positions, axis=0)
            self.board_visible = (len(hand_positions) >= 2)
            return self.last_hand_pos.copy(), markers_info

        self.board_visible = False
        return None, markers_info


# ============================================================
# PIPELINE STATE
# ============================================================
class PipelineState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.onset_detected  = False
        self.onset_time      = 0.0
        self.onset_pos       = None
        self.prediction_made = False
        self.last_pred_time  = 0.0

        self.result = {
            "label":      "--",
            "P_K1":       0.5,
            "P_K2":       0.5,
            "confidence": 0.0,
            "reason":     "waiting",
            "correct":    None,
        }

    @property
    def in_cooldown(self):
        return (time.time() - self.last_pred_time) < COOLDOWN


# ============================================================
# GLOBAL OBJECTS
# ============================================================
# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ArUco detector
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# BNO055
i2c    = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)
sensor.mode = adafruit_bno055.NDOF_MODE
time.sleep(0.5)
print("[HW] BNO055 initialised in NDOF mode")

# Wait for minimal calibration (gyro at least = 1)
print("[HW] Waiting for BNO055 gyro calibration (hold sensor still 5s)...")
for _ in range(50):
    cal = sensor.calibration_status
    if cal[1] >= 1:  # gyro cal
        break
    time.sleep(0.1)
print(f"[HW] Cal status (sys,gyro,accel,mag): {sensor.calibration_status}")

# Home position — set after first camera detection, use IMU until then
home_pos = np.array([0.0, 0.0, 1.0])  # assume 1m from board initially

# EKF
ekf   = EKF6(initial_pos=home_pos, dt=DT_IMU)
board_tracker = BoardTracker()
state = PipelineState()

# Stage 5 history buffer (circular, HISTORY_SIZE = 10 steps = 100ms at 100Hz)
pos_history  = np.tile(home_pos, (HISTORY_SIZE, 1))  # shape (10, 3)
step_counter = 0

# Timing
last_imu_time = time.time()
last_cam_time = time.time()
prev_quat     = np.array([1.0, 0.0, 0.0, 0.0])


# ============================================================
# IMU READER
# ============================================================
def read_bno055():
    """
    Stage 2: Read BNO055 outputs.
    Returns:
      a_lin:  linear acceleration (gravity-compensated, body frame) [m/s²]
      q:      absolute orientation quaternion [w, x, y, z]
      euler:  (heading, roll, pitch) in degrees
      cal:    (sys, gyro, accel, mag) calibration status 0-3
    """
    try:
        lin = sensor.linear_acceleration  # gravity-free, body frame
        quat = sensor.quaternion           # absolute orientation [w,x,y,z]
        euler = sensor.euler               # (heading, pitch, roll) degrees
        cal   = sensor.calibration_status

        a_lin = np.array(lin  if lin  and all(v is not None for v in lin)  else [0.0, 0.0, 0.0])
        q     = np.array(quat if quat and all(v is not None for v in quat) else [1.0, 0.0, 0.0, 0.0])

        heading = euler[0] if euler and euler[0] is not None else 0.0
        pitch   = euler[1] if euler and euler[1] is not None else 0.0
        roll    = euler[2] if euler and euler[2] is not None else 0.0

        return a_lin, q, (heading, roll, pitch), cal
    except Exception as e:
        print(f"[IMU] Read error: {e}")
        return np.zeros(3), np.array([1.0,0,0,0]), (0.0, 0.0, 0.0), (0,0,0,0)


# ============================================================
# STAGES 5 + 6 — FEATURE EXTRACTION + PROBABILISTIC INFERENCE
# ============================================================
def run_prediction(ekf_pos, ekf_vel, K1_world, K2_world):
    """
    Stage 5: Extract 7-dimensional feature vector F = [θ₁,θ₂,d₁,d₂,ḋ₁,ḋ₂,‖v‖]
    Stage 6: Softmax scoring → P(K1|F), P(K2|F)
    """
    global step_counter, pos_history

    v_mag = np.linalg.norm(ekf_vel)
    v_hat = ekf_vel / (v_mag + 1e-8)

    # Distances to each object
    d1 = np.linalg.norm(K1_world - ekf_pos)
    d2 = np.linalg.norm(K2_world - ekf_pos)

    # Distance rates from circular history buffer
    p_old = pos_history[(step_counter + 1) % HISTORY_SIZE]   # oldest entry
    d1_old = np.linalg.norm(K1_world - p_old)
    d2_old = np.linalg.norm(K2_world - p_old)
    dt_hist = HISTORY_SIZE * DT_IMU  # 0.10 s
    d1_dot  = (d1 - d1_old) / dt_hist
    d2_dot  = (d2 - d2_old) / dt_hist

    # Direction unit vectors toward each object
    d1_hat = (K1_world - ekf_pos) / (d1 + 1e-8)
    d2_hat = (K2_world - ekf_pos) / (d2 + 1e-8)

    # Angles between velocity and direction to each object
    theta_1 = np.arccos(np.clip(np.dot(v_hat, d1_hat), -1.0, 1.0))
    theta_2 = np.arccos(np.clip(np.dot(v_hat, d2_hat), -1.0, 1.0))

    # Stage 6: Logistic scoring with full feature vector
    s1 = (W_THETA * theta_1 + W_DDOT * d1_dot + W_DIST * d1) / TEMP
    s2 = (W_THETA * theta_2 + W_DDOT * d2_dot + W_DIST * d2) / TEMP

    # Numerically stable softmax
    ms = max(s1, s2)
    e1 = np.exp(s1 - ms)
    e2 = np.exp(s2 - ms)
    P1 = e1 / (e1 + e2)
    P2 = e2 / (e1 + e2)

    label      = "K1" if P1 > P2 else "K2"
    confidence = max(P1, P2) * 100.0

    return {
        "label":      label,
        "P_K1":       round(P1, 3),
        "P_K2":       round(P2, 3),
        "confidence": round(confidence, 1),
        "theta_1_deg": round(np.degrees(theta_1), 1),
        "theta_2_deg": round(np.degrees(theta_2), 1),
        "d1":          round(d1, 3),
        "d2":          round(d2, 3),
        "d1_dot":      round(d1_dot, 3),
        "d2_dot":      round(d2_dot, 3),
        "v_mag":       round(v_mag, 3),
        "reason":      f"θ₁={np.degrees(theta_1):.0f}° θ₂={np.degrees(theta_2):.0f}° → {label}",
    }


# ============================================================
# MAIN WEBSOCKET HANDLER
# ============================================================
async def handler(websocket):
    global step_counter, pos_history, ekf, state
    global last_imu_time, last_cam_time, home_pos

    print(f"[WS] Client connected: {websocket.remote_address}")

    # Reset pipeline on new connection
    state.reset()
    ekf = EKF6(initial_pos=home_pos, dt=DT_IMU)
    pos_history = np.tile(home_pos, (HISTORY_SIZE, 1))
    step_counter = 0

    try:
        while True:
            loop_start = time.time()

            # ── STAGE 2a: Read BNO055 ─────────────────────────────
            a_lin, q, euler, cal = read_bno055()
            heading, roll, pitch = euler
            cal_sys, cal_gyro, cal_accel, cal_mag = cal

            # ── STAGE 4: EKF Predict (100 Hz) ────────────────────
            ekf.predict(a_lin, q)

            # ── STAGE 2b: Camera frame ────────────────────────────
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # ── STAGE 1 + 4 Correct: Get hand position from board ─
            hand_pos_world, markers_info = board_tracker.process_frame(corners, ids)

            cam_corrected = False
            if hand_pos_world is not None:
                # STAGE 4 correction step: camera gives absolute hand position
                ekf.correct(hand_pos_world)
                cam_corrected = True
                # Update home position on first stable detection (hand at rest)
                if not state.onset_detected and not state.in_cooldown:
                    home_pos = ekf.get_pos().copy()
                    pos_history = np.tile(home_pos, (HISTORY_SIZE, 1))

            ekf_pos = ekf.get_pos()
            ekf_vel = ekf.get_vel()
            v_mag   = np.linalg.norm(ekf_vel)

            # Update history buffer
            pos_history[step_counter % HISTORY_SIZE] = ekf_pos.copy()
            step_counter += 1

            # ── STAGE 3: Motion Onset Detection ──────────────────
            now = time.time()
            if (not state.onset_detected
                    and not state.in_cooldown
                    and board_tracker.board_visible
                    and v_mag > ONSET_THRESH):
                state.onset_detected = True
                state.onset_time     = now
                state.onset_pos      = ekf_pos.copy()
                print(f"[ONSET] Detected! ‖v‖={v_mag:.3f} m/s  pos={ekf_pos.round(3)}")

            # ── STAGES 5+6: Predict at 300ms after onset ─────────
            if (state.onset_detected
                    and not state.prediction_made
                    and (now - state.onset_time) >= PREDICT_AT):

                result = run_prediction(
                    ekf_pos, ekf_vel,
                    board_tracker.K1_world,
                    board_tracker.K2_world
                )
                state.result = result
                state.prediction_made = True
                state.last_pred_time  = now

                print(f"[PRED] {result['label']}  "
                      f"P(K1)={result['P_K1']:.2f}  P(K2)={result['P_K2']:.2f}  "
                      f"conf={result['confidence']:.1f}%")

                # LEDs
                led_green.off()
                led_red.off()
                if result["label"] == "K1":
                    led_green.on()
                else:
                    led_red.on()

            # Reset cycle after cooldown
            if state.prediction_made and not state.in_cooldown:
                state.onset_detected  = False
                state.prediction_made = False
                led_green.off()
                led_red.off()

            # Also reset onset if hand stops (v_mag < thresh for >0.5s)
            if state.onset_detected and not state.prediction_made and v_mag < ONSET_THRESH * 0.5:
                state.onset_detected = False

            # ── Encode frame ──────────────────────────────────────
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            frame_b64 = base64.b64encode(buf).decode('utf-8')

            # ── Build payload ─────────────────────────────────────
            payload = json.dumps({
                "frame": frame_b64,
                "markers": markers_info,
                "imu": {
                    "heading": round(heading, 1),
                    "roll":    round(roll, 1),
                    "pitch":   round(pitch, 1),
                    "ax": round(float(a_lin[0]), 3),
                    "ay": round(float(a_lin[1]), 3),
                    "az": round(float(a_lin[2]), 3),
                    "cal_sys":   cal_sys,
                    "cal_gyro":  cal_gyro,
                    "cal_accel": cal_accel,
                    "cal_mag":   cal_mag,
                },
                "ekf": {
                    "px": round(float(ekf_pos[0]), 4),
                    "py": round(float(ekf_pos[1]), 4),
                    "pz": round(float(ekf_pos[2]), 4),
                    "vx": round(float(ekf_vel[0]), 4),
                    "vy": round(float(ekf_vel[1]), 4),
                    "vz": round(float(ekf_vel[2]), 4),
                    "v_mag":     round(v_mag, 4),
                    "onset":     state.onset_detected,
                    "board_vis": board_tracker.board_visible,
                    "cam_ok":    cam_corrected,
                    "ready":     board_tracker.board_visible and cal_sys >= 1,
                },
                "objects": {
                    "K1": board_tracker.K1_world.tolist(),
                    "K2": board_tracker.K2_world.tolist(),
                },
                "prediction": state.result,
            })

            await websocket.send(payload)

            # Sleep remaining budget (~33ms target = ~30Hz)
            elapsed = time.time() - loop_start
            sleep_t = max(0.0, 0.033 - elapsed)
            await asyncio.sleep(sleep_t)

    except websockets.exceptions.ConnectionClosed:
        print("[WS] Client disconnected")
    finally:
        led_green.off()
        led_red.off()


# ============================================================
# MAIN
# ============================================================
async def main():
    print("\n" + "=" * 60)
    print("  SmartGlove — 7-Stage Pipeline (Camera on Glove)")
    print("  Board: 4×ArUco 10cm markers, 30cm apart")
    print("  IMU:   BNO055 NDOF mode @ 100Hz")
    print("  ws://0.0.0.0:8765")
    print("=" * 60 + "\n")

    async with websockets.serve(
        handler, "0.0.0.0", 8765,
        max_size=5 * 1024 * 1024
    ):
        await asyncio.Future()


asyncio.run(main())

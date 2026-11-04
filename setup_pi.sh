#!/usr/bin/env bash
# =====================================================================
#  SmartGlove — one-shot installer for Raspberry Pi (Pi 5 / Bookworm)
#
#  Installs everything needed for Stream.py / Stream_nano.py:
#    system : python3-venv, i2c-tools, v4l-utils, lgpio (GPIO on Pi 5)
#    python : opencv-contrib (ArUco), numpy, websockets, pyserial,
#             adafruit-blinka + BNO055 driver, gpiozero
#    setup  : enables I2C, adds you to hardware groups, creates venv
#
#  Usage:
#      chmod +x setup_pi.sh
#      ./setup_pi.sh
#      sudo reboot            # first time only
#
#  Run your project:
#      cd <folder with Stream.py>
#      ~/smartglove/run.sh Stream.py
# =====================================================================
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/smartglove}"
VENV="$PROJECT_DIR/venv"

say()  { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
ok()   { printf "\033[1;32m[ OK ]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[FAIL]\033[0m %s\n" "$*"; exit 1; }

[ "$(id -u)" -eq 0 ] && die "Run as your normal user (not root / not with sudo). The script calls sudo itself."
command -v apt-get >/dev/null || die "This installer needs Raspberry Pi OS / Debian (apt)."

MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
IS_PI5=0; [[ "$MODEL" == *"Raspberry Pi 5"* ]] && IS_PI5=1
echo "Board: $MODEL"

# ---------------------------------------------------------------------
say "1/7  System packages (apt)"
sudo apt-get update || warn "apt update had errors (check internet)"
sudo apt-get install -y \
    python3 python3-venv python3-pip python3-dev build-essential \
    git i2c-tools v4l-utils libgl1 libglib2.0-0 \
    || die "apt install failed — check internet connection"
# GPIO backends (names differ by OS version, so each is optional)
for p in python3-lgpio python3-gpiozero python3-libgpiod; do
    sudo apt-get install -y "$p" >/dev/null 2>&1 && ok "$p" || warn "$p not available (skipped)"
done

# ---------------------------------------------------------------------
say "2/7  Enable I2C (for BNO055) and add hardware permissions"
if command -v raspi-config >/dev/null; then
    sudo raspi-config nonint do_i2c 0 && ok "I2C enabled"
else
    warn "raspi-config not found — enable I2C manually"
fi
for g in i2c gpio video dialout; do
    getent group "$g" >/dev/null && sudo usermod -aG "$g" "$USER" && ok "added $USER to group: $g"
done

# ---------------------------------------------------------------------
say "3/7  Python virtual environment  ($VENV)"
mkdir -p "$PROJECT_DIR"
# --system-site-packages lets the venv see apt's lgpio/gpiozero
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV" || die "venv creation failed"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools || die "pip upgrade failed"
ok "venv ready: $(python --version)"

# ---------------------------------------------------------------------
say "4/7  Python modules (pip)"
# opencv-contrib: needed for cv2.aruco.ArucoDetector (OpenCV >= 4.7).
# 'headless' = no GUI libs needed; your code never opens an OpenCV window.
pip install \
    numpy \
    opencv-contrib-python-headless \
    websockets \
    pyserial \
    gpiozero \
    adafruit-blinka \
    adafruit-circuitpython-bno055 \
    || die "pip install failed"
ok "core modules installed"

# ---------------------------------------------------------------------
say "5/7  Pi 5 GPIO fix"
if [ "$IS_PI5" -eq 1 ]; then
    # Pi 5 has a new GPIO chip: old RPi.GPIO does not work, rpi-lgpio replaces it
    pip uninstall -y RPi.GPIO >/dev/null 2>&1 || true
    pip install rpi-lgpio && ok "rpi-lgpio installed (Pi 5)" || warn "rpi-lgpio install failed"
else
    ok "Not a Pi 5 — nothing to do"
fi

# ---------------------------------------------------------------------
say "6/7  Creating run helper"
cat > "$PROJECT_DIR/run.sh" <<EOF
#!/usr/bin/env bash
# Usage: ~/smartglove/run.sh Stream.py
source "$VENV/bin/activate"
exec python "\${1:-Stream.py}"
EOF
chmod +x "$PROJECT_DIR/run.sh"
ok "created $PROJECT_DIR/run.sh"

# ---------------------------------------------------------------------
say "7/7  Verifying installation"
python - <<'PY'
import importlib, sys
mods = [("numpy","numpy"),("cv2","opencv"),("websockets","websockets"),
        ("serial","pyserial"),("gpiozero","gpiozero"),
        ("board","adafruit-blinka"),("busio","adafruit-blinka"),
        ("adafruit_bno055","adafruit-bno055")]
bad = 0
for m, name in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  [ OK ] {name:<18} {getattr(mod,'__version__','')}")
    except Exception as e:
        bad += 1
        print(f"  [FAIL] {name:<18} {type(e).__name__}: {e}")
try:
    import cv2
    assert hasattr(cv2.aruco, "ArucoDetector")
    print("  [ OK ] cv2.aruco.ArucoDetector available")
except Exception:
    bad += 1
    print("  [FAIL] cv2.aruco.ArucoDetector missing (need opencv-contrib >= 4.7)")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && ok "all Python modules import correctly" || warn "some modules failed — see above"

say "Hardware check (informational)"
echo "-- I2C (BNO055 should appear as 28 or 29):"
i2cdetect -y 1 2>/dev/null || echo "   I2C not active yet (reboot first, then re-run: i2cdetect -y 1)"
echo "-- Camera devices:";        ls /dev/video* 2>/dev/null || echo "   none found — plug in the USB camera"
echo "-- Arduino Nano (serial):"; ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "   none (only needed if using the Nano instead of BNO055)"

cat <<EOF

=====================================================================
 DONE.   >>> Run:  sudo reboot   (needed once: I2C + group changes)

 Then:
   cd <folder containing Stream.py and dashboard.html>
   ~/smartglove/run.sh Stream.py          # BNO055 version
   NANO_PORT=/dev/ttyACM0 ~/smartglove/run.sh Stream_nano.py   # Nano version

 Open dashboard.html in the Pi's browser (it connects to ws://localhost:8765).
 Optional: for better accuracy, calibrate the camera and save
   ~/camera_matrix.npy and ~/dist_coeffs.npy  (Stream.py loads them automatically).
=====================================================================
EOF

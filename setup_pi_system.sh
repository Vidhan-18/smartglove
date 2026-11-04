#!/usr/bin/env bash
# =====================================================================
#  SmartGlove — system-wide installer (NO virtual environment)
#  Raspberry Pi 5 / Raspberry Pi OS Bookworm
#
#  After this you can simply run:   python3 Stream.py
#
#  Usage:
#      chmod +x setup_pi_system.sh
#      ./setup_pi_system.sh
#      sudo reboot
# =====================================================================
set -uo pipefail

say()  { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
ok()   { printf "\033[1;32m[ OK ]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[FAIL]\033[0m %s\n" "$*"; exit 1; }

[ "$(id -u)" -eq 0 ] && die "Run as your normal user (not with sudo). The script calls sudo itself."
command -v apt-get >/dev/null || die "Needs Raspberry Pi OS / Debian."

MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
IS_PI5=0; [[ "$MODEL" == *"Raspberry Pi 5"* ]] && IS_PI5=1
echo "Board: $MODEL"

# ---------------------------------------------------------------------
say "1/6  System packages (apt)"
sudo apt-get update || warn "apt update had errors"
sudo apt-get install -y python3 python3-pip python3-dev build-essential \
    python3-numpy python3-serial python3-websockets python3-gpiozero python3-lgpio \
    i2c-tools v4l-utils libgl1 libglib2.0-0 \
    || die "apt install failed — check internet"
if [ "$IS_PI5" -eq 1 ]; then
    # Pi 5: RPi.GPIO does not work; rpi-lgpio is the drop-in replacement
    sudo apt-get install -y python3-rpi-lgpio >/dev/null 2>&1 \
        && ok "python3-rpi-lgpio (Pi 5 GPIO)" || warn "python3-rpi-lgpio not in apt (will try pip)"
fi

# ---------------------------------------------------------------------
say "2/6  Enable I2C + hardware permissions"
command -v raspi-config >/dev/null && sudo raspi-config nonint do_i2c 0 && ok "I2C enabled"
for g in i2c gpio video dialout; do
    getent group "$g" >/dev/null && sudo usermod -aG "$g" "$USER" && ok "group: $g"
done

# ---------------------------------------------------------------------
say "3/6  Python modules (pip, system-wide for this user)"
# --break-system-packages: Bookworm blocks plain pip otherwise.
# numpy<2 + opencv<4.12: keeps numpy 1.x so it matches the apt packages.
# opencv-contrib: provides cv2.aruco.ArucoDetector (needs OpenCV >= 4.7;
#                 apt's python3-opencv is only 4.6, too old for your code).
PIP="python3 -m pip install --user --break-system-packages --upgrade"
$PIP pip 2>/dev/null || true
$PIP "numpy<2" "opencv-contrib-python-headless<4.12" \
    || die "OpenCV install failed"
$PIP adafruit-blinka adafruit-circuitpython-bno055 \
    || die "Adafruit BNO055 install failed"
ok "pip modules installed"

# ---------------------------------------------------------------------
say "4/6  Pi 5 GPIO fix"
if [ "$IS_PI5" -eq 1 ]; then
    # Blinka may pull the old RPi.GPIO from pip, which shadows rpi-lgpio and breaks on Pi 5
    python3 -m pip uninstall -y --break-system-packages RPi.GPIO >/dev/null 2>&1 || true
    if ! python3 -c "import RPi.GPIO" >/dev/null 2>&1; then
        $PIP rpi-lgpio >/dev/null 2>&1 && ok "rpi-lgpio installed via pip" || warn "could not install rpi-lgpio"
    else
        ok "RPi.GPIO shim (rpi-lgpio) present"
    fi
else
    ok "Not a Pi 5 — nothing to do"
fi

# ---------------------------------------------------------------------
say "5/6  Verifying (using plain python3)"
python3 - <<'PY'
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
    print("  [FAIL] cv2.aruco.ArucoDetector missing")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && ok "everything imports with plain python3" || warn "some modules failed — see above"

say "6/6  Hardware check (informational)"
echo "-- I2C (BNO055 appears as 28 or 29):"
i2cdetect -y 1 2>/dev/null || echo "   I2C not active yet — reboot, then run: i2cdetect -y 1"
echo "-- Camera:";  ls /dev/video* 2>/dev/null || echo "   none found — plug in the USB camera"
echo "-- Nano serial:"; ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "   none (only needed for the Arduino Nano version)"

cat <<EOF

=====================================================================
 DONE.  Now run:  sudo reboot

 Then simply:
   cd ~/smartglove
   python3 Stream.py
=====================================================================
EOF

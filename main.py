import math
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BRUSH_SIZE = 40     # wiped line thickness, in pixels
REGROW = 0.003      # how fast fog returns per frame (0 = never)
SCALE = 4           # fog is computed at 1/SCALE resolution (higher = faster)
SMOOTHING = 0.5     # 1.0 = no smoothing, lower = smoother but more delay

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(options)


def is_pointing(hand):
    wrist = hand[0]

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def extended(tip, pip):
        return dist(wrist, hand[tip]) > dist(wrist, hand[pip]) * 1.1

    index_out = extended(8, 6)
    others_curled = not any(extended(t, p) for t, p in [(12, 10), (16, 14), (20, 18)])
    return index_out and others_curled


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("Could not open camera.")

WINDOW = "Foggy Mirror"
fog = None        # small float mask: 1.0 = fog, 0.0 = clear
prev_tips = {}    # last smoothed fingertip position per hand (full-res coords)
start = time.monotonic()
brush = max(1, BRUSH_SIZE // SCALE)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror
    h, w = frame.shape[:2]
    sw, sh = w // SCALE, h // SCALE

    if fog is None:
        fog = np.ones((sh, sw), dtype=np.float32)

    # Detect hands (video mode needs increasing timestamps in ms)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp = int((time.monotonic() - start) * 1000)
    result = detector.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
    )

    # Wipe fog along the smoothed fingertip path
    seen = set()
    for i, hand in enumerate(result.hand_landmarks):
        if not is_pointing(hand):
            continue
        seen.add(i)

        x, y = hand[8].x * w, hand[8].y * h
        if i in prev_tips:
            px, py = prev_tips[i]
            x = px + SMOOTHING * (x - px)
            y = py + SMOOTHING * (y - py)
            cv2.line(fog, (int(px / SCALE), int(py / SCALE)),
                     (int(x / SCALE), int(y / SCALE)), 0.0, brush)
        else:
            cv2.circle(fog, (int(x / SCALE), int(y / SCALE)), brush // 2, 0.0, -1)
        prev_tips[i] = (x, y)
    for i in list(prev_tips):
        if i not in seen:
            del prev_tips[i]

    # Fog slowly creeps back
    if REGROW > 0:
        fog += REGROW
        np.minimum(fog, 1.0, out=fog)

    # Foggy frame: blur + wash toward white, all at low resolution
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (13, 13), 0)
    small = cv2.convertScaleAbs(small, alpha=0.4, beta=153)  # 0.4*img + 0.6*white
    foggy = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Upscale the mask (this also smooths the wiped edges) and blend
    alpha = cv2.resize(cv2.GaussianBlur(fog, (5, 5), 0), (w, h),
                       interpolation=cv2.INTER_LINEAR)
    out = cv2.blendLinear(frame, foggy, 1.0 - alpha, alpha)

    cv2.imshow(WINDOW, out)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:            # Esc to quit
        break
    if key == ord("c"):      # C to refill the fog
        fog[:] = 1.0
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # X clicked
        break

cap.release()
cv2.destroyAllWindows()
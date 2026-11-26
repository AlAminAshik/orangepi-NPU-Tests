#view npu load live: watch sudo cat /sys/kernel/debug/rknpu/load
# Simple 3-core NPU detection: 3 copies of the model, each pinned to ONE NPU core.
# Frames are handed to the copies in turn, so the 3 cores work on 3 frames at once.

from ultralytics import YOLO          # import this BEFORE rknnlite
import cv2
import itertools
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from rknnlite.api import RKNNLite

# 1) Every model that gets loaded is pinned to the next core: 0, then 1, then 2.
cores = itertools.cycle([RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2])
original_init = RKNNLite.init_runtime

def init_one_core(self, *args, **kwargs):
    return original_init(self, *args, core_mask=next(cores), **kwargs)

RKNNLite.init_runtime = init_one_core

# 2) Three models, each with its own worker thread (one job at a time per model).
N = 3
models = [YOLO("orangepi-NPU-Tests/models/yolo11l_rknn_model_FP16", task="detect") for _ in range(N)]
workers = [ThreadPoolExecutor(max_workers=1) for _ in range(N)]

def detect(model, frame):
    # run detection and return the frame with the boxes drawn on it
    return model(frame, conf=0.3, imgsz=640, verbose=False)[0].plot()

# 3) Camera + loop
cap = cv2.VideoCapture(0)
pending = deque()          # frames being processed, oldest first
i = 0
count = 0
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    k = i % N                                          # frame 0 -> model 0, frame 1 -> model 1, ...
    pending.append(workers[k].submit(detect, models[k], frame))
    i += 1

    if len(pending) >= 2 * N:                          # keep up to 6 frames in flight
        cv2.imshow("Detection", pending.popleft().result())   # oldest first = correct order
        count += 1
        if count % 30 == 0:
            print(f"FPS: {30 / (time.time() - t0):.1f}")
            t0 = time.time()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

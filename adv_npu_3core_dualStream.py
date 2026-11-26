##view npu load live: watch sudo cat /sys/kernel/debug/rknpu/load
"""
TEST1:
rapoo and normal camera video as feed and the model is being is yolo11l, quantized INT8 rknn. The NPU utilization is approx 85% across the 3 core NPU.  the CPU usage are about 25-40% across all the 8 cores and RAM usage is 5GB.
the FPS I am getting is ~23 combined (~13.6 and ~9.8fps individually).
"""

"""
TEST2:
rapoo and normal camera video as feed and the model is being is yolo11n, not quantized FP16 rknn. The NPU utilization is approx 38% across the 3 core NPU.  the CPU usage are about 40-50% across all the 8 cores and RAM usage is 4.5GB.
the FPS I am getting is ~30 combined (~20 and ~9.8fps individually).
"""

"""
TEST3:
rapoo and normal camera video as feed and the model is being is yolo11n, quantized INT8 rknn. The NPU utilization is approx 21% across the 3 core NPU.  the CPU usage are about 25-40% across all the 8 cores and RAM usage is 4.5GB.
the FPS I am getting is ~29.9 combined (~20 and ~9.8fps individually).
"""

"""
TEST4:
rapoo and normal camera video as feed and the model is being is yolo11l, not quantized FP16 rknn. The NPU utilization is approx 87% across the 3 core NPU.  the CPU usage are about 20-30% across all the 8 cores and RAM usage is 5GB.
the FPS I am getting is ~10 combined (~4.1 and ~5.5fps individually).
"""

"""
Dual USB-camera YOLO11 on RK3588 using RKNNLite directly (no Ultralytics / torch at runtime).

- Each camera has its own reader thread; frames go to the NPU workers through shared memory.
- One worker process per NPU core, shared by both cameras.
- If the NPU falls behind, new camera frames are skipped (counted as "dropped") so the view stays live.
- Per-camera FPS is drawn on each pane; total FPS and NPU load in the bar below.
"""
import os
import glob
import time
import queue
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import deque

import cv2
import numpy as np

# ---------------- settings ----------------
CAMERAS = [0, 2]                    # /dev/videoN numbers (USB cams usually take 2 nodes each: 0, 2, 4...)
CAM_WIDTH, CAM_HEIGHT = 1280, 720   # requested capture size
CAM_FPS = 60                        # requested capture rate (camera must support it in MJPG)
MODEL_PATH = "yolo11l_rknn_model_INT8"   # folder from the Ultralytics export, or a .rknn file
CORES = [0, 1, 2]                   # NPU cores
CONTEXTS_PER_CORE = 1
CONF = 0.3
IOU = 0.7
MAX_DET = 300
IMGSZ = 640                         # must match the export imgsz
SLOTS = 16                          # shared-memory frame buffers
PANE_WIDTH = 640                    # display width of each pane
DISPLAY_MAX_FPS = 30                # window refresh cap (FPS figures count processed frames)
FPS_WINDOW = 60
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
          (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
          (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
          (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]
NUM_STREAMS = len(CAMERAS)
# ------------------------------------------


def find_rknn(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.rknn"))
        if not files:
            raise FileNotFoundError(f"No .rknn file in {path}")
        return files[0]
    return path


def load_names(path):
    meta = os.path.join(path if os.path.isdir(path) else os.path.dirname(path), "metadata.yaml")
    try:
        import yaml
        with open(meta) as f:
            names = yaml.safe_load(f).get("names", {})
        return {int(k): str(v) for k, v in names.items()}
    except Exception:
        return {}


# ---------------- NPU worker (unchanged) ----------------

def postprocess(output):
    p = output[0]
    if p.ndim == 3:
        p = p[0]
    if p.shape[0] > p.shape[1]:
        p = p.T
    scores = p[4:]
    conf = scores.max(0)
    keep = conf > CONF
    if not keep.any():
        return np.zeros((0, 6), np.float32)
    conf = conf[keep]
    cls = scores[:, keep].argmax(0)
    b = p[:4, keep].T.astype(np.float32)
    if b.max() <= 1.5:
        b *= IMGSZ
    x1 = b[:, 0] - b[:, 2] / 2
    y1 = b[:, 1] - b[:, 3] / 2
    nms_boxes = np.stack([x1 + cls * 4096.0, y1, b[:, 2], b[:, 3]], 1)
    idx = cv2.dnn.NMSBoxes(nms_boxes.tolist(), conf.tolist(), CONF, IOU)
    idx = np.array(idx, dtype=np.int64).flatten()[:MAX_DET]
    dets = np.stack([x1, y1, x1 + b[:, 2], y1 + b[:, 3], conf, cls.astype(np.float32)], 1)
    return dets[idx].astype(np.float32)


def worker(core_id, rknn_file, shm_name, in_q, out_q, ready_q):
    from rknnlite.api import RKNNLite
    cv2.setNumThreads(1)

    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray((SLOTS, IMGSZ, IMGSZ, 3), np.uint8, buffer=shm.buf)

    rknn = RKNNLite(verbose=False)
    mask = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2][core_id]
    if rknn.load_rknn(rknn_file) != 0 or rknn.init_runtime(core_mask=mask) != 0:
        ready_q.put((core_id, f"could not load {rknn_file} on NPU core {core_id}"))
        return
    out = rknn.inference(inputs=[buf[0][None]], data_format="nhwc")
    ready_q.put((core_id, None, [o.shape for o in out]))

    while True:
        job = in_q.get()
        if job is None:
            break
        stream, idx, slot = job
        out = rknn.inference(inputs=[buf[slot][None]], data_format="nhwc")
        out_q.put((stream, idx, slot, postprocess(out)))

    rknn.release()
    shm.close()


# ---------------- cameras ----------------

def open_camera(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # set the format before the size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def put_until(q, item, stop):
    while not stop.is_set():
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            pass
    return False


def camera_reader(stream, cap, in_q, free_slots, buf, slot_geom, slot_refs, frames, state, dropped, stop):
    """Reads one camera, letterboxes each frame into shared memory and queues it for the NPU."""
    idx = 0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            print(f"Camera {CAMERAS[stream]} stopped delivering frames")
            break
        try:
            slot = free_slots.get_nowait()
        except queue.Empty:
            dropped[stream] += 1          # NPU busy: skip this frame so the view stays live
            continue

        h, w = frame.shape[:2]
        r = min(IMGSZ / h, IMGSZ / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        px, py = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
        dst = buf[slot]
        if slot_geom[slot] != (nw, nh):
            dst[:] = 114
            slot_geom[slot] = (nw, nh)
        resized = frame if (nw, nh) == (w, h) else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dst[py:py + nh, px:px + nw] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        pane_scale = PANE_WIDTH / w
        pane = cv2.resize(frame, (PANE_WIDTH, int(round(h * pane_scale))), interpolation=cv2.INTER_LINEAR)
        frames[(stream, idx)] = (pane, px, py, pane_scale / r)

        slot_refs[slot] = 1
        if not put_until(in_q, (stream, idx, slot), stop):
            break
        idx += 1
    cap.release()
    state[stream] = idx


# ---------------- display ----------------

def draw_dets(img, dets, px, py, k, names):
    for x1, y1, x2, y2, conf, c in dets:
        c = int(c)
        color = COLORS[c % len(COLORS)]
        p1 = (int((x1 - px) * k), int((y1 - py) * k))
        p2 = (int((x2 - px) * k), int((y2 - py) * k))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{names.get(c, c)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        ty = max(p1[1], th + 4)
        cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), color, -1)
        cv2.putText(img, label, (p1[0] + 2, ty - 3), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_labels(img, lines, color=(0, 255, 0)):
    y = 8
    for text in lines:
        (w, h), base = cv2.getTextSize(text, FONT, 0.7, 2)
        cv2.rectangle(img, (8, y), (8 + w + 12, y + h + base + 10), (0, 0, 0), -1)
        cv2.putText(img, text, (14, y + h + 5), FONT, 0.7, color, 2, cv2.LINE_AA)
        y += h + base + 14


def stream_fps(t, start, shown):
    cur = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0
    avg = (shown - 1) / (t[-1] - start) if shown > 1 and t[-1] > start else 0.0
    return cur, avg


def read_npu_load():
    try:   # needs root (debugfs)
        with open("/sys/kernel/debug/rknpu/load") as f:
            return f.read().strip()
    except OSError:
        return None


def render(latest, times, start, shown, dropped, npu_text, names):
    panes, total = [], 0.0
    for s, ((pane, px, py, k), dets) in enumerate(latest):
        img = pane.copy()
        draw_dets(img, dets, px, py, k, names)
        cur, avg = stream_fps(times[s], start[s], shown[s])
        total += cur
        draw_labels(img, [f"Cam {s + 1} (/dev/video{CAMERAS[s]})   FPS: {cur:.1f}   avg: {avg:.1f}",
                          f"skipped: {dropped[s]}"])
        panes.append(img)
    tallest = max(p.shape[0] for p in panes)   # cameras may differ in aspect ratio
    panes = [cv2.copyMakeBorder(p, 0, tallest - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panes]
    view = cv2.hconcat(panes)
    bar = np.zeros((40, view.shape[1], 3), np.uint8)
    status = f"Total: {total:.1f} FPS"
    if npu_text:
        status += "    " + npu_text
    cv2.putText(bar, status, (10, 27), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("YOLO11 RKNN - dual camera", cv2.vconcat([view, bar]))


# ---------------- main ----------------

def main():
    rknn_file = find_rknn(MODEL_PATH)
    names = load_names(MODEL_PATH)

    # open the cameras first so a wrong device number fails immediately
    caps = []
    for dev in CAMERAS:
        cap = open_camera(dev)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open /dev/video{dev} - check `v4l2-ctl --list-devices`")
        print(f"/dev/video{dev}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
        caps.append(cap)

    ctx = mp.get_context("spawn")
    shm = shared_memory.SharedMemory(create=True, size=SLOTS * IMGSZ * IMGSZ * 3)
    buf = np.ndarray((SLOTS, IMGSZ, IMGSZ, 3), np.uint8, buffer=shm.buf)
    buf[:] = 114

    n_workers = len(CORES) * CONTEXTS_PER_CORE
    in_q = ctx.Queue(maxsize=n_workers * 2)
    out_q = ctx.Queue()
    ready_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(core, rknn_file, shm.name, in_q, out_q, ready_q), daemon=True)
             for core in CORES for _ in range(CONTEXTS_PER_CORE)]

    stop = threading.Event()
    readers = []
    try:
        for p in procs:
            p.start()
        for _ in procs:
            msg = ready_q.get(timeout=120)
            if msg[1] is not None:
                raise RuntimeError(msg[1])
            print(f"NPU core {msg[0]} ready, output shape(s): {msg[2]}")

        free_slots = queue.Queue()
        for i in range(SLOTS):
            free_slots.put(i)
        slot_refs = [0] * SLOTS
        slot_geom = [None] * SLOTS
        frames, state = {}, {}
        dropped = [0] * NUM_STREAMS
        readers = [threading.Thread(target=camera_reader,
                                    args=(s, caps[s], in_q, free_slots, buf, slot_geom, slot_refs,
                                          frames, state, dropped, stop), daemon=True)
                   for s in range(NUM_STREAMS)]
        for t in readers:
            t.start()

        next_show = [0] * NUM_STREAMS
        pending = [{} for _ in range(NUM_STREAMS)]
        latest = [None] * NUM_STREAMS
        times = [deque(maxlen=FPS_WINDOW + 1) for _ in range(NUM_STREAMS)]
        start = [None] * NUM_STREAMS
        shown = [0] * NUM_STREAMS
        oldest = [0] * NUM_STREAMS
        last_draw = 0.0
        refresh = 1.0 / DISPLAY_MAX_FPS if DISPLAY_MAX_FPS else 0.0
        npu_text, npu_checked = None, 0.0

        while True:
            results = []
            try:
                results.append(out_q.get(timeout=0.005))
                while True:
                    results.append(out_q.get_nowait())
            except queue.Empty:
                pass

            now = time.time()
            for s, idx, slot, dets in results:
                pending[s][idx] = dets
                slot_refs[slot] -= 1
                if slot_refs[slot] == 0:
                    free_slots.put(slot)

            for s in range(NUM_STREAMS):
                while next_show[s] in pending[s]:
                    latest[s] = (frames[(s, next_show[s])], pending[s].pop(next_show[s]))
                    next_show[s] += 1
                    shown[s] += 1
                    times[s].append(now)
                    if start[s] is None:
                        start[s] = now
                while oldest[s] < next_show[s]:          # drop display frames already shown
                    frames.pop((s, oldest[s]), None)
                    oldest[s] += 1

            if now - npu_checked > 0.5:
                npu_text, npu_checked = read_npu_load(), now

            # stop if a camera was unplugged and everything it sent has been shown
            if any(s in state and next_show[s] >= state[s] for s in range(NUM_STREAMS)):
                print("A camera stopped; exiting.")
                break

            if now - last_draw >= refresh:
                if all(l is not None for l in latest):
                    render(latest, times, start, shown, dropped, npu_text, names)
                    last_draw = now
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        stop.set()
        for t in readers:
            t.join(timeout=2)
        for cap in caps:
            cap.release()
        try:
            while True:
                in_q.get_nowait()
        except queue.Empty:
            pass
        for _ in procs:
            try:
                in_q.put(None, timeout=1)
            except queue.Full:
                pass
        for p in procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        cv2.destroyAllWindows()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()

#view npu load live: watch sudo cat /sys/kernel/debug/rknpu/load
"""
TEST1:
60fps video as feed and the model is being is yolo11n, quantized to INT8 rknn. The NPU utilization is approx 40-50% across the 3 core NPU.  the CPU usage are about 40-60% across all the 8 cores and RAM usage is 4GB.
the FPS I am getting is ~55 combined (~27 and ~27fps individually).
"""

"""
TEST2:
60fps video as feed and the model is being is yolo11l, quantized to INT8 rknn. The NPU utilization is approx 90% across the 3 core NPU.  the CPU usage are about 30-40% across all the 8 cores and RAM usage is 4GB.
the FPS I am getting is ~24 combined (~11.5 and ~11.5fps individually).
"""

"""
TEST3:
60fps video as feed and the model is being is yolo11n, not quantized, FP16 rknn. The NPU utilization is approx 60% across the 3 core NPU.  the CPU usage are about 60-70% across all the 8 cores and RAM usage is 4GB.
the FPS I am getting is ~43 combined (~20.6 and ~20.6fps individually).
"""

"""
Dual-stream YOLO11 benchmark on RK3588 using RKNNLite directly (no Ultralytics / torch at runtime).

- The video is decoded ONCE; every frame is run through the model NUM_STREAMS times.
- One worker process per NPU core; frames reach the workers through shared memory (no pickling).
- Workers return only the detections; drawing happens on small display-sized frames.
- Per-stream FPS is drawn on each pane; total FPS and NPU load in the bar below.
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
VIDEO_PATH = "60fpsTestVideo.mp4"
MODEL_PATH = "yolo11n_rknn_model_INT8"   # folder from the Ultralytics export, or a .rknn file
NUM_STREAMS = 2                     # how many times each frame is run through the model
CORES = [0, 1, 2]                   # NPU cores
CONTEXTS_PER_CORE = 1               # 1 is right for yolo11n; try 2 for large models
CONF = 0.3
IOU = 0.7                           # same default as Ultralytics
MAX_DET = 300
IMGSZ = 640                         # must match the export imgsz
SLOTS = 16                          # shared-memory frame buffers
PANE_WIDTH = 640                    # display width of each pane
DISPLAY_MAX_FPS = 30                # window refresh cap (FPS figures count processed frames, not refreshes)
REALTIME = True                    # True = feed at the video's own frame rate; False = as fast as possible
LOOP_VIDEO = True                  # restart the video automatically at EOF
USE_HW_DECODE = False              # True = GStreamer decodebin (uses mppvideodec if gst-rockchip is installed)
FPS_WINDOW = 60
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
          (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
          (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
          (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]
# ------------------------------------------


def find_rknn(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.rknn"))
        if not files:
            raise FileNotFoundError(f"No .rknn file in {path}")
        return files[0]
    return path


def load_names(path):
    """Class names from the metadata.yaml that Ultralytics writes next to the .rknn file."""
    meta = os.path.join(path if os.path.isdir(path) else os.path.dirname(path), "metadata.yaml")
    try:
        import yaml
        with open(meta) as f:
            names = yaml.safe_load(f).get("names", {})
        return {int(k): str(v) for k, v in names.items()}
    except Exception:
        return {}


# ---------------- NPU worker ----------------

def postprocess(output):
    """YOLO11 detect output (1, 4+nc, 8400) -> array of [x1, y1, x2, y2, conf, cls] in 640x640 letterbox space."""
    p = output[0]
    if p.ndim == 3:
        p = p[0]
    if p.shape[0] > p.shape[1]:          # (8400, 4+nc) -> (4+nc, 8400)
        p = p.T
    scores = p[4:]
    conf = scores.max(0)
    keep = conf > CONF
    if not keep.any():
        return np.zeros((0, 6), np.float32)
    conf = conf[keep]
    cls = scores[:, keep].argmax(0)
    b = p[:4, keep].T.astype(np.float32)  # cx, cy, w, h
    if b.max() <= 1.5:                    # some INT8 exports output normalised boxes
        b *= IMGSZ
    x1 = b[:, 0] - b[:, 2] / 2
    y1 = b[:, 1] - b[:, 3] / 2
    # class-aware NMS: shift each class far apart so boxes of different classes never overlap
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
    out = rknn.inference(inputs=[buf[0][None]], data_format="nhwc")   # warm-up
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


# ---------------- decoding ----------------

def open_video():
    if USE_HW_DECODE:
        pipe = (f'filesrc location="{VIDEO_PATH}" ! decodebin ! videoconvert ! '
                f'video/x-raw,format=BGR ! appsink sync=false max-buffers=4')
        return cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(VIDEO_PATH)


def put_until(q, item, stop):
    while not stop.is_set():
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            pass
    return False


def get_until(q, stop):
    while not stop.is_set():
        try:
            return q.get(timeout=0.2)
        except queue.Empty:
            pass
    return None


def decoder(in_q, free_slots, buf, slot_refs, frames, state, stop):
    """Decodes each frame once, letterboxes it into shared memory, and queues it for every stream."""
    cap = open_video()
    if not cap.isOpened():
        print(f"Could not open {VIDEO_PATH}")
        state["total"] = 0
        return
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    slot_geom = [None] * SLOTS
    t_start = time.time()
    idx = 0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            if not LOOP_VIDEO or stop.is_set():
                break
            # Rewind on EOF. Reopen if seeking is unsupported by the decoder.
            if cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = open_video()
                if not cap.isOpened():
                    print(f"Could not reopen {VIDEO_PATH} for looping")
                    break
                ok, frame = cap.read()
            if not ok:
                print(f"Could not read the first frame of {VIDEO_PATH}")
                break
            # Keep idx increasing across loops for worker ordering and FPS timing.
        if REALTIME:
            delay = t_start + idx / video_fps - time.time()
            if delay > 0:
                time.sleep(delay)

        slot = get_until(free_slots, stop)
        if slot is None:
            break

        # letterbox straight into the shared buffer (RGB, uint8, grey padding)
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

        # small copy for display; k maps letterbox coords -> pane coords
        pane_scale = PANE_WIDTH / w
        pane = cv2.resize(frame, (PANE_WIDTH, int(round(h * pane_scale))), interpolation=cv2.INTER_LINEAR)
        frames[idx] = (pane, px, py, pane_scale / r)

        slot_refs[slot] = NUM_STREAMS
        for s in range(NUM_STREAMS):
            if not put_until(in_q, (s, idx, slot), stop):
                break
        idx += 1
    cap.release()
    state["total"] = idx


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


def render(latest, times, start, shown, npu_text, names):
    panes, total, total_avg = [], 0.0, 0.0
    for s, ((pane, px, py, k), dets) in enumerate(latest):
        img = pane.copy()
        draw_dets(img, dets, px, py, k, names)
        cur, avg = stream_fps(times[s], start[s], shown[s])
        total += cur
        total_avg += avg
        draw_labels(img, [f"Stream {s + 1}   FPS: {cur:.1f}   avg: {avg:.1f}"])
        panes.append(img)
    view = cv2.hconcat(panes)
    bar = np.zeros((40, view.shape[1], 3), np.uint8)
    status = f"Total: {total:.1f} FPS (avg {total_avg:.1f})"
    if npu_text:
        status += "    " + npu_text
    cv2.putText(bar, status, (10, 27), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("YOLO11 RKNN - dual stream", cv2.vconcat([view, bar]))


# ---------------- main ----------------

def main():
    rknn_file = find_rknn(MODEL_PATH)
    names = load_names(MODEL_PATH)

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
    dec = None
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
        frames, state = {}, {}
        dec = threading.Thread(target=decoder, args=(in_q, free_slots, buf, slot_refs, frames, state, stop),
                               daemon=True)
        dec.start()

        next_show = [0] * NUM_STREAMS
        pending = [{} for _ in range(NUM_STREAMS)]
        latest = [None] * NUM_STREAMS
        times = [deque(maxlen=FPS_WINDOW + 1) for _ in range(NUM_STREAMS)]
        start = [None] * NUM_STREAMS
        shown = [0] * NUM_STREAMS
        oldest = 0
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
                if slot_refs[slot] == 0:          # every stream has used this frame
                    free_slots.put(slot)

            for s in range(NUM_STREAMS):
                while next_show[s] in pending[s]:  # keep each stream in order
                    latest[s] = (frames[next_show[s]], pending[s].pop(next_show[s]))
                    next_show[s] += 1
                    shown[s] += 1
                    times[s].append(now)
                    if start[s] is None:
                        start[s] = now

            lowest = min(next_show)                # drop display frames no stream needs any more
            while oldest < lowest:
                frames.pop(oldest, None)
                oldest += 1

            if now - npu_checked > 0.5:
                npu_text, npu_checked = read_npu_load(), now

            finished = "total" in state and all(n >= state["total"] for n in next_show)
            if finished and any(l is None for l in latest):
                break

            if finished or now - last_draw >= refresh:
                if all(l is not None for l in latest):
                    render(latest, times, start, shown, npu_text, names)
                    last_draw = now
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if finished:
                cv2.waitKey(0)   # keep the final frame and FPS on screen until a key press
                break
    finally:
        stop.set()
        if dec is not None:
            dec.join(timeout=2)
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

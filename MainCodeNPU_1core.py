#view npu load live: watch sudo cat /sys/kernel/debug/rknpu/load
# One rapoo camera was used for this test.
# inference is about 55ms about 18fps.
# this code uses only one core of NPU at about 35%


from ultralytics import YOLO
import cv2
import pygame
import numpy as np

# Load model
model = YOLO("orangepi-NPU-Tests/models/bestN_rknn_model_FP16", task="detect")

cap = cv2.VideoCapture(0)  # webcam

# Initialize pygame mixer
pygame.mixer.init()

def beep():
    freq = 1000  # Hz
    duration = 200  # ms
    sample_rate = 44100
    n_samples = int(round(duration * sample_rate / 1000))
    t = np.linspace(0, duration / 1000, n_samples, False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)

    sound_array = np.int16(wave * 32767)
    sound_array = np.column_stack((sound_array, sound_array))  # stereo

    sound = pygame.sndarray.make_sound(sound_array)
    sound.play()   # async (non-blocking)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3, imgsz=640)
    detected_labels = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = model.names[cls]
            detected_labels.append(label)

            # For 960x540 resolution
            box_thickness = 12         # bounding box thickness
            font_scale = 2.0           # font size ~2x bigger
            font_thickness = 6         # bold text

            # Draw detection box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 255, 0), box_thickness)

            # Put text
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 20),   # shift text higher
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 0),
                        font_thickness)

    if detected_labels:
        # Beep when detected
        beep()

        # Show prominent label at the top
        text = "Detect: " + ", ".join(set(detected_labels))

        # Get text size (so we can draw background)
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            3.2,  # same scale
            6     # same thickness
        )

        # Background rectangle (red)
        cv2.rectangle(
            frame,
            (50, 100 - text_h - baseline),
            (50 + text_w, 100 + baseline),
            (0, 0, 255),  # red (BGR)
            -1            # filled
        )

        # Draw text (white)
        cv2.putText(
            frame,
            text,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.2,
            (255, 255, 255),  # white
            6
        )

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()

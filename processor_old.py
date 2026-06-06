import cv2
import numpy as np


def detect_peaks(signal, min_dist, threshold):
    peaks = []
    last = -min_dist

    for i in range(1, len(signal) - 1):
        if (
            signal[i] > threshold and
            signal[i] > signal[i - 1] and
            signal[i] > signal[i + 1] and
            i - last >= min_dist
        ):
            peaks.append(i)
            last = i

    return peaks

def process_video(video_path, roi, accelerated, progress_cb, frame_cb, cancel_flag):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x, y, w, h = roi

    intensity = []
    frame_idx = 0

    while True:
        if cancel_flag():
            cap.release()
            return None

        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        intensity.append(np.mean(gray))

        frame_idx += 1
        if progress_cb and total_frames > 0:
            progress_cb(int(100 * frame_idx / total_frames))

        if not accelerated and frame_cb:
            frame_cb(frame.copy(), roi)

    cap.release()

    if len(intensity) == 0:
        raise RuntimeError("No se obtuvieron datos del ROI")

    signal = np.array(intensity)
    signal -= signal.mean()
    signal = (signal - signal.min()) / (signal.max() - signal.min())

    peaks = detect_peaks(
        signal,
        min_dist=int(fps / 4),
        threshold=0.1
    )

    duration = len(signal) / fps
    bpm = len(peaks) * (60 / duration)
    return bpm, signal, peaks, fps

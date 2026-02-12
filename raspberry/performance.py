import argparse
import time
import os
from collections import deque
from multiprocessing import Process, Event, Lock

import numpy as np
import pandas as pd
import serial
import tflite_runtime.interpreter as tflite

from parse_data import from_buffer_to_df_detection
from features_raspberry import extract_features

# ================= CONFIG =================
N_SAMPLES_SCREEN_UPDATE = 1
MOVING_AVG_SIZE_PREDICTIONS = 10
DATA_COLUMNS_NAMES = ["type", "local_timestamp", "data"]
CSI_DATA_LENGTH = 384
SAMPLING_FREQUENCY = 20
SEGMENTATION_WINDOW_LENGTH = 200
RECEIVE_TIMEOUT = 2
CSI_RX_BAUDRATE = 460800
LCD_BAUDRATE = 115200
MODEL_PATH = f"models/csi_hr_best_{SEGMENTATION_WINDOW_LENGTH}.tflite"

# ================= HELPER =================
def track_time(stats, duration):
    stats['count'] += 1
    stats['total'] += duration
    stats['max'] = max(stats['max'], duration)

def print_stats(name, stats):
    if stats['count'] > 0:
        avg = stats['total'] / stats['count']
        print(f"[{name}] Avg: {avg*1000:.2f} ms | Max: {stats['max']*1000:.2f} ms | Count: {stats['count']}")

# ================= PROCESSES =================
def csi_read_process(port, q_out, lock, stop_event):
    os.sched_setaffinity(0, {0})
    print(f"Worker PID {os.getpid()} assegnato al core 0")
    stats = {'count':0, 'total':0.0, 'max':0.0}

    try:
        ser = serial.Serial(port, CSI_RX_BAUDRATE, timeout=RECEIVE_TIMEOUT)
        print("CSI receiver: open success")
    except OSError:
        print("CSI receiver: open failed")
        stop_event.set()
        return

    time.sleep(2)
    ser.reset_input_buffer()
    ser.write(b"START\n")

    while not stop_event.is_set():
        start = time.perf_counter()
        strings = str(ser.readline())
        if not strings:
            continue
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        with lock:
            q_out.append(strings)
        track_time(stats, time.perf_counter() - start)

    try:
        ser.write(b"STOP\n")
        time.sleep(1)
        ser.close()
    except:
        pass
    print_stats("CSI_READ", stats)


def csi_process_process(q_in, q_out, lock_in, lock_out, stop_event):
    os.sched_setaffinity(0, {1})
    print(f"Worker PID {os.getpid()} assegnato al core 1")
    current_df = pd.DataFrame(columns=DATA_COLUMNS_NAMES)
    stats = {'count':0, 'total':0.0, 'max':0.0}

    while not stop_event.is_set():
        buffer = None
        with lock_in:
            if len(q_in) > 0:
                buffer = list(q_in)
                q_in.clear()
        if buffer is None:
            continue

        start = time.perf_counter()
        df = from_buffer_to_df_detection(buffer, DATA_COLUMNS_NAMES)
        if df.empty:
            continue

        current_df = pd.concat([current_df, df], ignore_index=True)
        if len(current_df) < SEGMENTATION_WINDOW_LENGTH:
            continue

        window = extract_features(
            current_df.head(SEGMENTATION_WINDOW_LENGTH),
            CSI_DATA_LENGTH,
            SAMPLING_FREQUENCY,
            SEGMENTATION_WINDOW_LENGTH
        )
        current_df = current_df.iloc[1:].reset_index(drop=True)

        if window is not None:
            with lock_out:
                q_out.append(window)
        track_time(stats, time.perf_counter() - start)

    print_stats("CSI_PROCESS", stats)


def prediction_process(q_in, q_out, lock_in, lock_out, stop_event):
    os.sched_setaffinity(0, {2})
    print(f"Worker PID {os.getpid()} assegnato al core 2")
    stats = {'count':0, 'total':0.0, 'max':0.0}

    print("Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded!")

    while not stop_event.is_set():
        window = None
        with lock_in:
            if len(q_in) > 0:
                window = q_in.popleft()
        if window is None:
            continue

        start = time.perf_counter()
        input_data = np.array(window, dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0][0]

        with lock_out:
            if len(q_out) >= q_out.maxlen:
                q_out.popleft()
            q_out.append(pred)
        track_time(stats, time.perf_counter() - start)

    print_stats("PREDICTION", stats)


def lcd_process(port, q_in, lock_in, stop_event):
    os.sched_setaffinity(0, {3})
    print(f"Worker PID {os.getpid()} assegnato al core 3")
    stats = {'count':0, 'total':0.0, 'max':0.0}

    try:
        ser = serial.Serial(port, LCD_BAUDRATE)
        print("STM32 lcd: open success")
    except OSError:
        print("STM32 lcd: open failed")
        stop_event.set()
        return

    samples = deque(maxlen=MOVING_AVG_SIZE_PREDICTIONS)
    n = 0
    ser.write(f"000\n".encode("ascii"))
    ser.flush()

    while not stop_event.is_set():
        pred = None
        with lock_in:
            if len(q_in) > 0:
                pred = q_in.popleft()
        if pred is None:
            continue

        start = time.perf_counter()
        samples.append(pred)
        n += 1
        avg = int(np.mean(samples))
        print(avg)

        if n % N_SAMPLES_SCREEN_UPDATE == 0:
            try:
                ser.write(f"{avg:03d}\n".encode("ascii"))
                ser.flush()
            except:
                pass
        track_time(stats, time.perf_counter() - start)

    try:
        ser.close()
    except:
        pass
    print_stats("LCD_PROCESS", stats)


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser(description="Start heartbeat sensing")
    parser.add_argument("-p", "--port", required=True, help="CSI receiver serial port")
    parser.add_argument("-ps", "--port_screen", required=True, help="LCD serial port")
    args = parser.parse_args()

    stop_event = Event()

    # deque con maxlen
    q_raw = deque(maxlen=200)
    q_window = deque(maxlen=50)
    q_pred = deque(maxlen=50)

    # lock per ogni deque
    lock_raw = Lock()
    lock_window = Lock()
    lock_pred = Lock()

    processes = [
        Process(target=csi_read_process, args=(args.port, q_raw, lock_raw, stop_event)),
        Process(target=csi_process_process, args=(q_raw, q_window, lock_raw, lock_window, stop_event)),
        Process(target=prediction_process, args=(q_window, q_pred, lock_window, lock_pred, stop_event)),
        Process(target=lcd_process, args=(args.port_screen, q_pred, lock_pred, stop_event)),
    ]

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Closing...")
        stop_event.set()
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()

import argparse
import time
import os
from collections import deque
from multiprocessing import Process, Queue, Event

import numpy as np
import pandas as pd
import serial
import tflite_runtime.interpreter as tflite  # type: ignore

from parse_data import from_buffer_to_df_detection
from features_raspberry import extract_features

import queue


N_SAMPLES_SCREEN_UPDATE = 1
MOVING_AVG_SIZE_PREDICTIONS = 5

DATA_COLUMNS_NAMES = ["type", "local_timestamp", "data"]

CSI_DATA_LENGTH = 384
SAMPLING_FREQUENCY = 20
SEGMENTATION_WINDOW_LENGTH = 100
RECEIVE_TIMEOUT = 2

CSI_RX_BAUDRATE = 460800
LCD_BAUDRATE = 115200

MODEL_PATH = f"models/csi_hr_best_{SEGMENTATION_WINDOW_LENGTH}.tflite"

MIN_SAMPLES_FOR_PROCESSING = 10
SCALING_MEAN = 77.683
SCALING_STD = 14.341


def safe_put(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


def get_all(q):
    items = []
    while len(items) < MIN_SAMPLES_FOR_PROCESSING:
        try:
            items.append(q.get())
        except:
            break
    return items


def report(times, name):
    if len(times) == 0:
        return
    avg = sum(times) / len(times)
    mx = max(times)
    print(f"{name} avg={avg:.6f}s max={mx:.6f}s")


def csi_read_process(port, q_out, stop_event):
    os.sched_setaffinity(0, {0})
    print(f"Worker PID {os.getpid()} assegnato al core {0}")

    ser = None
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

    times = []

    print("csi_read_process: Gathering data...")
    while not stop_event.is_set():
        strings = str(ser.readline())
        if not strings:
            continue
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        t0 = time.perf_counter()
        try:
            safe_put(q_out, strings)
        except:
            pass
        times.append(time.perf_counter() - t0)

        report(times, "csi_read_process")

    try:
        ser.write(b"STOP\n")
        time.sleep(1)
        ser.close()
    except:
        pass


def csi_process_process(q_in, q_out, stop_event):
    os.sched_setaffinity(0, {1})
    print(f"Worker PID {os.getpid()} assegnato al core {1}")

    current_df = pd.DataFrame(columns=DATA_COLUMNS_NAMES)
    times = []

    print("csi_process_process: waiting for data...")
    while not stop_event.is_set():
        try:
            buffer = get_all(q_in)
        except:
            continue

        t0 = time.perf_counter()

        df = from_buffer_to_df_detection(buffer, DATA_COLUMNS_NAMES)
        if df.empty:
            continue

        current_df = pd.concat([current_df, df], ignore_index=True)
        current_df_len = len(current_df)

        if current_df_len < SEGMENTATION_WINDOW_LENGTH:
            continue

        window = extract_features(
            current_df.head(SEGMENTATION_WINDOW_LENGTH),
            CSI_DATA_LENGTH,
            SAMPLING_FREQUENCY,
            SEGMENTATION_WINDOW_LENGTH
        )

        current_df = current_df.iloc[-SEGMENTATION_WINDOW_LENGTH:, :].reset_index(drop=True)

        if window is not None:
            try:
                safe_put(q_out, window)
            except:
                pass

        times.append(time.perf_counter() - t0)

        report(times, "csi_process_process")


def prediction_process(q_in, q_out, stop_event):
    os.sched_setaffinity(0, {2})
    print(f"Worker PID {os.getpid()} assegnato al core {2}")

    print("Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite model loaded!")
    print("prediction_process: waiting for window...")

    times = []

    while not stop_event.is_set():
        try:
            window = q_in.get(timeout=0.5)
        except:
            continue

        t0 = time.perf_counter()

        input_data = np.array(window, dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0][0]
        pred = (pred * SCALING_STD) + SCALING_MEAN

        try:
            safe_put(q_out, pred)
        except:
            pass

        times.append(time.perf_counter() - t0)

        report(times, "prediction_process")


def lcd_process(port, q_in, stop_event):
    os.sched_setaffinity(0, {3})
    print(f"Worker PID {os.getpid()} assegnato al core {3}")

    try:
        ser = serial.Serial(port, LCD_BAUDRATE)
        print("STM32 lcd: open success")
    except OSError:
        print("STM32 lcd: open failed")
        stop_event.set()
        return

    samples = deque(maxlen=MOVING_AVG_SIZE_PREDICTIONS)
    n = 0
    times = []

    print("lcd_process: waiting for prediction...")
    ser.write(f"000\n".encode("ascii"))
    ser.flush()

    while not stop_event.is_set():
        try:
            pred = q_in.get(timeout=0.5)
        except:
            continue

        t0 = time.perf_counter()

        samples.append(pred)
        n += 1
        avg = int(np.mean(samples))

        if n % N_SAMPLES_SCREEN_UPDATE == 0:
            try:
                ser.write(f"{avg:03d}\n".encode("ascii"))
                ser.flush()
            except:
                pass

        times.append(time.perf_counter() - t0)

        report(times, "lcd_process")

    try:
        ser.close()
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="Start heartbeat sensing")
    parser.add_argument("-p", "--port", required=True, help="CSI receiver serial port")
    parser.add_argument("-ps", "--port_screen", required=True, help="LCD serial port")

    args = parser.parse_args()

    stop_event = Event()

    q_raw = Queue(maxsize=200)
    q_window = Queue(maxsize=50)
    q_pred = Queue(maxsize=50)

    processes = [
        Process(target=csi_read_process, args=(args.port, q_raw, stop_event)),
        Process(target=csi_process_process, args=(q_raw, q_window, stop_event)),
        Process(target=prediction_process, args=(q_window, q_pred, stop_event)),
        Process(target=lcd_process, args=(args.port_screen, q_pred, stop_event)),
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

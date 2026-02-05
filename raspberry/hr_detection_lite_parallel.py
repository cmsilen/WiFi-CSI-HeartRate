import argparse
import time
from collections import deque
from multiprocessing import Process, Queue, Event

import numpy as np
import pandas as pd
import serial
import tflite_runtime.interpreter as tflite

from parse_data import iterate_data_rcv, from_buffer_to_df_detection
from features_raspberry import extract_features


# ================= CONFIG =================
N_SAMPLES_SCREEN_UPDATE = 10
MOVING_AVG_SIZE_PREDICTIONS = 10

DATA_COLUMNS_NAMES = ["type", "local_timestamp", "data"]

CSI_DATA_LENGTH = 384
SAMPLING_FREQUENCY = 20
SEGMENTATION_WINDOW_LENGTH = 200
RECEIVE_TIMEOUT = 2

CSI_RX_BAUDRATE = 921600
LCD_BAUDRATE = 115200


def rst_esp(ser):
    # Induce reset via DTR/RTS
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)  # breve pausa
    ser.dtr = True
    ser.rts = False

    # Ora la scheda è resettata
    print("ESP32 resettato")

    time.sleep(0.1)  # breve pausa
    ser.reset_input_buffer()

# ================= PROCESSES =================

def csi_read_process(port, q_out, stop_event):
    """Read CSI strings from serial and push to queue."""

    ser = None
    try:
        ser = serial.Serial(port, CSI_RX_BAUDRATE, timeout=RECEIVE_TIMEOUT)
        print("CSI receiver: open success")
    except OSError:
        print("CSI receiver: open failed")
        stop_event.set()
        return
    
    rst_esp(ser)

    ser.write(b"START\n")
    ser.reset_input_buffer()

    print("csi_read_process: Gathering data...")

    while not stop_event.is_set():
        strings = iterate_data_rcv(ser)
        if strings is None:
            continue

        try:
            q_out.put(strings, timeout=0.5)
        except:
            pass

    try:
        ser.write(b"STOP\n")
        time.sleep(1)
        ser.close()
    except:
        pass



def csi_process_process(q_in, q_out, stop_event):
    """Convert raw CSI to feature windows."""

    settings = {
        "training_phase": False,
        "verbose": False,
        "csi_data_length": CSI_DATA_LENGTH,
        "sampling_frequency": SAMPLING_FREQUENCY,
        "segmentation_window_length": SEGMENTATION_WINDOW_LENGTH,
    }

    current_df = pd.DataFrame(columns=DATA_COLUMNS_NAMES)

    print("csi_process_process: waiting for data...")

    while not stop_event.is_set():
        try:
            buffer = q_in.get(timeout=0.5)
        except:
            continue

        df = from_buffer_to_df_detection([buffer], DATA_COLUMNS_NAMES)
        if df.empty:
            continue

        current_df = pd.concat([current_df, df], ignore_index=True)

        if len(current_df) < SEGMENTATION_WINDOW_LENGTH:
            continue

        window = extract_features(
            current_df.head(SEGMENTATION_WINDOW_LENGTH), settings
        )

        current_df = current_df.iloc[1:].reset_index(drop=True)

        if window is not None:
            try:
                q_out.put(window, timeout=0.5)
            except:
                pass



def prediction_process(q_in, q_out, stop_event):
    """Run TFLite inference on windows."""

    print("Loading TFLite model...")

    interpreter = tflite.Interpreter(
        model_path=f"models/csi_hr_best_{SEGMENTATION_WINDOW_LENGTH}.tflite"
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite model loaded!")
    print("prediction_process: waiting for window...")

    while not stop_event.is_set():
        try:
            window = q_in.get(timeout=0.5)
        except:
            continue

        input_data = np.array(window, dtype=np.float32)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0][0]

        try:
            q_out.put(pred, timeout=0.5)
        except:
            pass



def lcd_process(port, q_in, stop_event):
    """Send averaged BPM predictions to LCD via serial."""

    try:
        ser = serial.Serial(port, LCD_BAUDRATE)
        print("STM32 lcd: open success")
    except OSError:
        print("STM32 lcd: open failed")
        stop_event.set()
        return

    samples = deque(maxlen=MOVING_AVG_SIZE_PREDICTIONS)
    n = 0

    print("lcd_process: waiting for prediction...")

    while not stop_event.is_set():
        try:
            pred = q_in.get(timeout=0.5)
        except:
            continue

        samples.append(pred)
        n += 1

        avg = int(np.mean(samples))
        print(avg)

        if n % N_SAMPLES_SCREEN_UPDATE == 0:
            try:
                ser.write(f"{avg:03d}\n".encode())
            except:
                pass

    try:
        ser.close()
    except:
        pass


# ================= MAIN =================

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

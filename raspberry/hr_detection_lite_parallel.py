import argparse
import time
import os
from collections import deque
from multiprocessing import Process, Queue, Event

import numpy as np
import pandas as pd
import serial
import tflite_runtime.interpreter as tflite

from parse_data import from_buffer_to_df_detection
from features_raspberry import extract_features

import queue


# ================= CONFIG =================
N_SAMPLES_SCREEN_UPDATE = 1                                             # how many predictions to do before updating the screen
MOVING_AVG_SIZE_PREDICTIONS = 10                                        # size of the moving average for the predictions

DATA_COLUMNS_NAMES = ["type", "local_timestamp", "data"]                 # data transferred via uart

CSI_DATA_LENGTH = 384                                                   # length of the csi data array (192 subcarriers x I/Q components)
SAMPLING_FREQUENCY = 20                                                 # frequency of csi sampling
SEGMENTATION_WINDOW_LENGTH = 200                                        # size of the LSTM window
RECEIVE_TIMEOUT = 2                                                     # maximum waiting time for data to arrive in the uart line

CSI_RX_BAUDRATE = 460800                                                # baud rate for the csi receiver (esp32)
LCD_BAUDRATE = 115200                                                   # baud rate for the lcd screen (stm32f4-discovery)

MODEL_PATH = f"models/csi_hr_best_{SEGMENTATION_WINDOW_LENGTH}.tflite"  # path of the model to be loaded

MIN_SAMPLES_FOR_PROCESSING = 10

def safe_put(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()  # rimuovi il più vecchio
        except queue.Empty:
            pass
        q.put_nowait(item)

def get_all(q):
    """Ritorna tutti gli elementi presenti nella coda al momento della chiamata."""
    items = []
    while len(items) < MIN_SAMPLES_FOR_PROCESSING:
        try:
            items.append(q.get())
        except:
            break  # in caso qualcuno abbia preso l'elemento prima
    return items

# ================= PROCESSES =================

def csi_read_process(port, q_out, stop_event):
    """Read CSI strings from serial and push to queue."""

    # assign the process to the core 0
    os.sched_setaffinity(0, {0})
    print(f"Worker PID {os.getpid()} assegnato al core {0}")

    # open port associated to the CSI receiver
    ser = None
    try:
        ser = serial.Serial(port, CSI_RX_BAUDRATE, timeout=RECEIVE_TIMEOUT)
        print("CSI receiver: open success")
    except OSError:
        print("CSI receiver: open failed")
        stop_event.set()
        return
    
    # wait for serial to be stable, clear eventual spurious bytes, send start command
    time.sleep(2)
    ser.reset_input_buffer()
    ser.write(b"START\n")

    # start receiving data
    print("csi_read_process: Gathering data...")
    while not stop_event.is_set():
        # read line and clear it
        strings = str(ser.readline())
        if not strings:
            print("no string from csi port")
            continue
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        # print(strings)

        # enqueue string
        try:
            safe_put(q_out, strings)
        except:
            pass

    try:
        ser.write(b"STOP\n")
        time.sleep(1)
        ser.close()
    except:
        pass



def csi_process_process(q_in, q_out, stop_event):
    """Convert raw CSI string to feature windows."""

    # assign the process to core 1
    os.sched_setaffinity(0, {1})
    print(f"Worker PID {os.getpid()} assegnato al core {1}")

    # dataframe containing the current window data
    current_df = pd.DataFrame(columns=DATA_COLUMNS_NAMES)

    print("csi_process_process: waiting for data...")
    while not stop_event.is_set():
        # dequeue string
        try:
            buffer = get_all(q_in) #[q_in.get(timeout=0.5)]
        except:
            continue

        # parse received string
        df = from_buffer_to_df_detection(buffer, DATA_COLUMNS_NAMES)
        if df.empty:
            continue

        # concat current data with the new one arrived
        current_df = pd.concat([current_df, df], ignore_index=True)
        current_df_len = len(current_df)

        # gather at least SEGMENTATION_WINDOW_LENGTH samples before extracting features
        if current_df_len < SEGMENTATION_WINDOW_LENGTH:
            print(f"gathering data... {current_df_len}/{SEGMENTATION_WINDOW_LENGTH}")
            continue

        # extract features
        window = extract_features(
            current_df.head(SEGMENTATION_WINDOW_LENGTH),
            CSI_DATA_LENGTH,
            SAMPLING_FREQUENCY,
            SEGMENTATION_WINDOW_LENGTH
        )

        # remove oldest csi data
        current_df = current_df.iloc[-SEGMENTATION_WINDOW_LENGTH:, :].reset_index(drop=True)

        # enqueue features
        if window is not None:
            try:
                safe_put(q_out, window)
            except:
                pass



def prediction_process(q_in, q_out, stop_event):
    """Run TFLite inference on windows."""

    # assign the process to core 2
    os.sched_setaffinity(0, {2})
    print(f"Worker PID {os.getpid()} assegnato al core {2}")

    # load model
    print("Loading TFLite model...")
    interpreter = tflite.Interpreter(
        model_path=MODEL_PATH
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite model loaded!")
    print("prediction_process: waiting for window...")

    while not stop_event.is_set():
        # dequeue features
        try:
            window = q_in.get(timeout=0.5)
        except:
            continue

        # perform inference
        input_data = np.array(window, dtype=np.float32)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0][0]

        # enqueue prediction
        try:
            safe_put(q_out, pred)
        except:
            pass



def lcd_process(port, q_in, stop_event):
    """Send averaged BPM predictions to LCD via serial."""

    # assign process to core 3
    os.sched_setaffinity(0, {3})
    print(f"Worker PID {os.getpid()} assegnato al core {3}")

    # open port associated to the lcd screen
    try:
        ser = serial.Serial(port, LCD_BAUDRATE)
        print("STM32 lcd: open success")
    except OSError:
        print("STM32 lcd: open failed")
        stop_event.set()
        return

    # moving average samples
    samples = deque(maxlen=MOVING_AVG_SIZE_PREDICTIONS)
    n = 0

    print("lcd_process: waiting for prediction...")
    ser.write(f"000\n".encode("ascii"))
    ser.flush()

    while not stop_event.is_set():
        # dequeue prediction
        try:
            pred = q_in.get(timeout=0.5)
        except:
            continue

        # add new samples to the others
        samples.append(pred)
        n += 1

        # moving average computation
        avg = int(np.mean(samples))
        print(avg)

        # update the lcd screen every N_SAMPLES_SCREEN_UPDATE predictions
        if n % N_SAMPLES_SCREEN_UPDATE == 0:
            try:
                ser.write(f"{avg:03d}\n".encode("ascii"))
                ser.flush()
            except:
                pass

    try:
        ser.close()
    except:
        pass


# ================= MAIN =================

def main():
    # get arguments
    parser = argparse.ArgumentParser(description="Start heartbeat sensing")
    parser.add_argument("-p", "--port", required=True, help="CSI receiver serial port")
    parser.add_argument("-ps", "--port_screen", required=True, help="LCD serial port")

    args = parser.parse_args()

    # stop event initialization
    stop_event = Event()

    # initialize all queues
    q_raw = Queue(maxsize=200)
    q_window = Queue(maxsize=50)
    q_pred = Queue(maxsize=50)

    # initialize processes
    processes = [
        Process(target=csi_read_process, args=(args.port, q_raw, stop_event)),
        Process(target=csi_process_process, args=(q_raw, q_window, stop_event)),
        Process(target=prediction_process, args=(q_window, q_pred, stop_event)),
        Process(target=lcd_process, args=(args.port_screen, q_pred, stop_event)),
    ]

    # start processes
    for p in processes:
        p.start()

    # wait for processes to end or ctrl-c
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

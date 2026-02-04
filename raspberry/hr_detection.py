import argparse
import numpy as np
import pandas as pd
import serial
import threading
import time
from collections import deque
from parse_data import iterate_data_rcv, from_buffer_to_df_detection
from features_raspberry import extract_features
from tensorflow import keras
import tensorflow as tf

N_SAMPLES_SCREEN_UPDATE = 10
MOVING_AVG_SIZE_PREDICTIONS = 10
DATA_COLUMNS_NAMES = ["type", "local_timestamp", "data"]
HR_COLUMNS_NAMES = ["IR", "BPM", "AVG BPM"]
CSI_DATA_LENGTH = 384               # esp32 exposes only 192 subcarriers, each carrier has associated I/Q components, so 192 x 2 = 384
SAMPLING_FREQUENCY = 20
SEGMENTATION_WINDOW_LENGTH = 200    # WINDOW_SIZE
RECEIVE_TIMEOUT = 2                 # seconds

CSI_RX_BAUDRATE = 921600
LCD_BAUDRATE = 115200


stop_event = threading.Event()

new_data_event = threading.Event()
buffer_csi = []
buffer_csi_lock = threading.Lock()

new_input_event = threading.Event()
input_lstm = None
input_lstm_lock = threading.Lock()

new_hr_event = threading.Event()
new_hr = None
new_hr_lock = threading.Lock()

tf.keras.mixed_precision.set_global_policy('mixed_float16')
model = keras.models.load_model(f"models/csi_hr_best_{SEGMENTATION_WINDOW_LENGTH}.keras", safe_mode=False)

ON_RPI = False
try:
    with open("/proc/device-tree/model") as f:
        ON_RPI = "raspberry pi" in f.read().lower()
except:
    pass

if ON_RPI:
    N_SAMPLES_SCREEN_UPDATE = 1

def csi_read_thread(port):
    global new_data_event
    global buffer_csi
    global stop_event
    ser = None
    try:
        ser = serial.Serial(port=port, baudrate=CSI_RX_BAUDRATE,bytesize=8, parity='N', stopbits=1, timeout=RECEIVE_TIMEOUT)
        if ser.isOpen():
            print("CSI receiver: open success")
        else:
            print("CSI receiver: open failed")
            stop_event.set()
            return
    except OSError:
        print("CSI receiver: open failed")
        stop_event.set()
        return
    
    # send start command
    string_start = "START\n"
    ser.write(string_start.encode("ascii"))
    ser.flush()
    ser.reset_input_buffer()
    print(f"Gathering data...")
    while not stop_event.is_set():
        strings = iterate_data_rcv(ser)
        if strings is None:
            break

        with buffer_csi_lock:
            buffer_csi.append(strings)
        
        new_data_event.set()
    
    # stop reception, clear buffer, close serial
    string_stop = "STOP\n"
    ser.write(string_stop.encode("ascii"))
    ser.flush()
    time.sleep(1)
    ser.reset_input_buffer()
    ser.close()

def csi_process_thread():
    global input_lstm
    global new_input_event
    global new_data_event
    global buffer_csi
    global buffer_csi_lock
    global stop_event
    settings = {}
    settings["training_phase"] = False
    settings["verbose"] = False
    settings["csi_data_length"] = CSI_DATA_LENGTH
    settings["sampling_frequency"] = SAMPLING_FREQUENCY
    settings["segmentation_window_length"] = SEGMENTATION_WINDOW_LENGTH
    current_df = pd.DataFrame(columns=DATA_COLUMNS_NAMES)

    while not stop_event.is_set():
        new_data_event.wait()
        new_data_event.clear()
        if stop_event.is_set():
            break

        with buffer_csi_lock:
            buffer = buffer_csi
            buffer_csi = []
        
        df = from_buffer_to_df_detection(buffer, DATA_COLUMNS_NAMES)
        if not df.empty:
            current_df = pd.concat([current_df, df], ignore_index=True)
        else:
            continue

        current_df_len = len(current_df)
        if current_df_len < SEGMENTATION_WINDOW_LENGTH:
            print(f"Gathering initial data: {current_df_len}/{SEGMENTATION_WINDOW_LENGTH}")
            continue
        
        window = extract_features(current_df.head(SEGMENTATION_WINDOW_LENGTH), settings)
        if window is None or len(window) != 1:
            continue
        current_df = current_df.iloc[1:].reset_index(drop=True)

        with input_lstm_lock:
            input_lstm = window.copy()
        new_input_event.set()


def prediction_thread():
    global input_lstm
    global new_input_event
    global stop_event
    global model
    global new_hr_lock
    global new_hr
    global new_hr_event
    
    while not stop_event.is_set():
        new_input_event.wait()
        new_input_event.clear()
        if stop_event.is_set():
            break

        with input_lstm_lock:
            window = input_lstm.copy()
        
        new_prediction = model.predict(window, verbose=0)
        prediction = new_prediction[0][0]
        with new_hr_lock:
            new_hr = prediction
        new_hr_event.set()


def lcd_thread(port):
    global new_hr
    global new_hr_event
    global new_hr_lock

    n_predictions = 0
    ser_screen = serial.Serial(port=port, baudrate=LCD_BAUDRATE,bytesize=8, parity='N', stopbits=1)
    samples = deque(maxlen=MOVING_AVG_SIZE_PREDICTIONS)
    if ser_screen.isOpen():
        print("STM32 lcd: open success")
    else:
        print("STM32 lcd: open failed")
        stop_event.set()
        return
    
    while not stop_event.is_set():
        new_hr_event.wait()
        new_hr_event.clear()
        if stop_event.is_set():
            break

        prediction = None
        with new_hr_lock:
            prediction = new_hr
        
        samples.append(prediction)
        n_predictions += 1
        avg_bpm = np.mean(samples)
        print(avg_bpm)

        # send to stm32
        if n_predictions % N_SAMPLES_SCREEN_UPDATE == 0:
            value = int(avg_bpm)
            string_s = f"{value:03d}\n"
            ser_screen.write(string_s.encode("ascii"))
    
    ser_screen.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Start hearbeat sensing")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of the receiver device")
    parser.add_argument('-ps', '--port_screen', dest='port_screen', action='store', required=True,
                        help="Serial port number of the screen device")
    

    args = parser.parse_args()
    serial_port = args.port
    serial_port_screen = args.port_screen
    
    t_read = threading.Thread(target=csi_read_thread, args=(serial_port,))
    t_process = threading.Thread(target=csi_process_thread)
    t_pred = threading.Thread(target=prediction_thread)
    t_lcd = threading.Thread(target=lcd_thread, args=(serial_port_screen,))

    t_read.start()
    t_process.start()
    t_pred.start()
    t_lcd.start()
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        print("Closing...")
        stop_event.set()
        new_data_event.set()
        new_input_event.set()
        new_hr_event.set()
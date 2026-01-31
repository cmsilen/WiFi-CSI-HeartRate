import argparse
import numpy as np
import pandas as pd
import serial
import threading
from collections import deque
from collect_data import iterate_data_rcv, from_buffer_to_df_detection
from features_train import extract_features
from tensorflow import keras


N_SAMPLES_SCREEN_UPDATE = 10
MOVING_AVG_SIZE_PREDICTIONS = 10
DATA_COLUMNS_NAMES = ["type", "local_timestamp", "data"]
HR_COLUMNS_NAMES = ["IR", "BPM", "AVG BPM"]
CSI_DATA_LENGTH = 384               # esp32 exposes only 192 subcarriers, each carrier has associated I/Q components, so 192 x 2 = 384
SAMPLING_FREQUENCY = 20
SEGMENTATION_WINDOW_LENGTH = 200    # WINDOW_SIZE


stop_event = threading.Event()

new_data_event = threading.Event()
buffer_csi = []
buffer_csi_lock = threading.Lock()

new_input_event = threading.Event()
input_lstm = None
input_lstm_lock = threading.Lock()

model = keras.models.load_model(f"models/csi_hr_best_{SEGMENTATION_WINDOW_LENGTH}.keras", safe_mode=False)

def csi_read_thread(port):
    global new_data_event
    global buffer_csi
    global stop_event
    ser = serial.Serial(port=port, baudrate=115200,bytesize=8, parity='N', stopbits=1)
    if ser.isOpen():
        print("open success")
    else:
        print("open failed")
        return
    
    # send start command
    string_start = "START\n"
    ser.write(string_start.encode("ascii"))
    print(f"Gathering data...")
    while not stop_event.is_set():
        outcome, strings, _ = iterate_data_rcv(ser, None, None, None, True)
        if outcome is None:
            break
        if outcome == False:
            continue

        with buffer_csi_lock:
            buffer_csi.append(strings)
        
        new_data_event.set()
    
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

        with buffer_csi_lock:
            buffer = buffer_csi.copy()
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
        current_df = current_df.iloc[1:].reset_index(drop=True)
        if len(window) != 1:
            continue

        with input_lstm_lock:
            input_lstm = window.copy()
        new_input_event.set()


def prediction_thread(port):
    global input_lstm
    global new_input_event
    global stop_event

    n_predictions = 0
    ser_screen = serial.Serial(port=port, baudrate=115200,bytesize=8, parity='N', stopbits=1)
    samples = deque(maxlen=MOVING_AVG_SIZE_PREDICTIONS)
    if ser_screen.isOpen():
        print("open success")
    else:
        print("open failed")
        return
    
    while not stop_event.is_set():
        new_input_event.wait()
        new_input_event.clear()

        with input_lstm_lock:
            window = input_lstm.copy()
        
        new_prediction = model.predict(window, verbose=0)
        prediction = new_prediction[0][0]
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
    t_pred = threading.Thread(target=prediction_thread, args=(serial_port_screen,))

    t_read.start()
    t_process.start()
    t_pred.start()
    try:
        t_read.join()
        stop_event.set()
        t_process.join()
        stop_event.set()
        t_pred.join()
    except KeyboardInterrupt:
        print("Closing...")
        stop_event.set()
import sys
import os
import csv
import json
import argparse
import numpy as np
import serial
from io import StringIO
import ast
from scipy.signal import butter, filtfilt, savgol_filter
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import re

ACCEPTED_HR_RANGE = [60, 90]
ACCEPTED_IR_RANGE = [100000, 120000]
DATA_COLUMNS_NAMES = ["type", "local_timestamp", "data"]
HR_COLUMNS_NAMES = ["IR", "BPM", "AVG BPM"]


def parse_csi_line(line, cols):
    result = None
    try:
        result = pd.read_csv(
            StringIO(line),
            names=cols,
            quotechar='"'
        )
    except pd.errors.ParserError:
        result = None
    return result

def safe_parse_csi_data(data_str):
    if not isinstance(data_str, str):
        return None

    match = re.search(r"\[.*\]", data_str)
    if not match:
        return None

    try:
        return ast.literal_eval(match.group())
    except Exception:
        return None

def iterate_data_rcv(ser, ser_hr, ir_range, avg_bpm_range, print_bpm=False):
    strings = str(ser.readline())
    string_hr = None
    if ser_hr is not None:
        string_hr = str(ser_hr.readline())
    if not strings:
        print("no string from csi port")
        return None, None, None
    if not string_hr and ser_hr is not None:
        print("no string from hr port")
        return None, None, None
    strings = strings.lstrip('b\'').rstrip('\\r\\n\'')

    if ser_hr is not None:
        string_hr = string_hr.lstrip('b\'').rstrip('\\r\\n\'')
        split_hr = string_hr.split(",")
        avg_bpm_str = split_hr[2]
        ir = split_hr[0]
        if not ir.isdigit():
            print(f"no ir: {ir}")
            return False, None, None
        if not avg_bpm_str.isdigit():
            print(f"no hr: {avg_bpm_str}")
            return False, None, None
        ir_int = int(ir)
        avg_bpm = avg_bpm_str
        avg_bpm_int = int(avg_bpm)
        if ir_int < ir_range[0] or ir_int > ir_range[1]:
            print(f"invalid ir: {ir_int}")
            return False, None, None
        if avg_bpm_int < avg_bpm_range[0] or avg_bpm_int > avg_bpm_range[1]:
            print(f"invalid hr: {avg_bpm_int}")
            return False, None, None
        
        if print_bpm:
            print(f"Current bpm: {avg_bpm}")
    return True, strings, string_hr


def from_buffer_to_df(buffer_csi, buffer_hr, cols_csi, cols_hr, csi_data_length=384):
    df_csi = pd.DataFrame(columns=cols_csi)
    df_hr = pd.DataFrame(columns=cols_hr)
    for line_csi, line_hr in zip(buffer_csi, buffer_hr):
        result_csi = parse_csi_line(line_csi, cols_csi)
        result_hr = parse_csi_line(line_hr, cols_hr)
        if result_csi is None or result_hr is None:
            continue
        df_csi = pd.concat([df_csi] + [result_csi], ignore_index=True)
        df_hr = pd.concat([df_hr] + [result_hr], ignore_index=True)
    
    df = pd.concat([df_csi.reset_index(drop=True), df_hr.reset_index(drop=True)], axis=1)
    df = df[df["type"] == "CSI_DATA"].copy()
    df["csi_raw"] = df["data"].apply(safe_parse_csi_data)
    df = df.dropna()
    df["csi_len"] = df["csi_raw"].apply(len)
    df = df[df["csi_len"] == csi_data_length].copy()
    return df

import pandas as pd

def from_buffer_to_df_detection(buffer_csi, cols_csi, csi_data_length=384):
    parsed_rows = []

    for line_csi in buffer_csi:
        result_csi = parse_csi_line(line_csi, cols_csi)
        if result_csi is not None:
            # Converti il risultato (DataFrame a riga singola) in dizionario e aggiungi alla lista
            parsed_rows.append(result_csi.iloc[0].to_dict())

    # Crea il DataFrame in un colpo solo
    df_csi = pd.DataFrame(parsed_rows, columns=cols_csi)

    # Filtra solo CSI_DATA
    df_csi = df_csi[df_csi["type"] == "CSI_DATA"].copy()
    df_csi["csi_raw"] = df_csi["data"].apply(safe_parse_csi_data)
    df_csi = df_csi.dropna()
    df_csi["csi_len"] = df_csi["csi_raw"].apply(len)
    df_csi = df_csi[df_csi["csi_len"] == csi_data_length].copy()

    return df_csi


def csi_data_read_parse(port: str, port_hr: str):
    global fft_gains, agc_gains
    ser = serial.Serial(port=port, baudrate=115200,bytesize=8, parity='N', stopbits=1)
    ser_hr = serial.Serial(port=port_hr, baudrate=115200,bytesize=8, parity='N', stopbits=1)
    if ser.isOpen() and ser_hr.isOpen():
        print("open success")
    else:
        print("open failed")
        return
    
    buffer_csi = []
    buffer_hr = []
    try:
        # send start command
        string_start = "START\n"
        ser.write(string_start.encode("ascii"))
        while True:
            outcome, strings, string_hr = iterate_data_rcv(ser, ser_hr, ACCEPTED_IR_RANGE, ACCEPTED_HR_RANGE, True)
            if outcome is None:
                break
            if outcome == False:
                continue
            buffer_csi.append(strings)
            buffer_hr.append(string_hr)

    except KeyboardInterrupt:
        print("Saving data...")
    
    ser.close()
    ser_hr.close()

    df = from_buffer_to_df(buffer_csi, buffer_hr, DATA_COLUMNS_NAMES, HR_COLUMNS_NAMES)
    df = df[["local_timestamp", "csi_raw", "IR", "BPM", "AVG BPM"]]
    df.to_csv("data/raw_data.csv", mode="a", header=not os.path.exists("data/raw_data.csv"), index=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device")
    parser.add_argument('-phr', '--port_hr', dest='port_hr', action='store', required=True,
                        help="Serial port number of csv_recv device")

    args = parser.parse_args()
    serial_port = args.port
    serial_port_hr = args.port_hr

    csi_data_read_parse(serial_port, serial_port_hr)
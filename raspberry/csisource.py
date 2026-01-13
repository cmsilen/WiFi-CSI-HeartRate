import csv
import serial
from io import StringIO
from collections import deque

from log import print_log, LVL_DBG, LVL_INF, LVL_ERR

# Serial port number
port = None

# Indicates wheter to get data from serial or csv
from_serial = True

# Serial connection
ser = None

# Csv list
csv_array = []
csv_len = 0

# Current csv index
csv_index = 0

def csi_data_source_init(port, from_ser: bool = True):
    """
    Initializes the source of the data (serial or csv)
    
    :param port: Serial port or csv filename
    :param from_ser: Indicates wheter to get data from serial or csv
    :type from_ser: bool
    """
    global ser, csv_array, from_serial, csv_len

    from_serial = from_ser

    if from_serial:
        ser = serial.Serial(port=port, baudrate=921600, timeout=1)
        if ser.isOpen():
            print_log("Serial port opened successfully", LVL_INF)
            print_log(f"Port     = {port}", LVL_DBG)
            print_log(f"Baudrate = {921600}", LVL_DBG)
            # print_log(f"Bytesize = {8}", LVL_DBG)
            # print_log(f"Parity   = {'N'}", LVL_DBG)
            # print_log(f"stopbits = {1}", LVL_DBG)
        else:
            print("Serial port - Open failed", LVL_ERR)
            return -1
        
    else:
        with open(port, newline='') as csvfile:
            full_data = list(csv.reader(csvfile, delimiter=','))
            for data_row in full_data:
                csv_array.append(data_row)

            csv_len = len(full_data)
                
            print_log(f"CSV lines read: {csv_len}", LVL_DBG)

def csi_data_source_close(dbg_print: bool = False):
    """
    Closes the csi data source
    
    :param dbg_print: Indicates wheter to print debug information
    :type dbg_print: bool
    """

    global from_serial, ser
    
    if from_serial:
        ser.close()
        if dbg_print:
            print_log("Serial closed successfully", LVL_INF)


def get_csi_data():
    global from_serial, ser, csv_array, csv_index, csv_len

    if from_serial:
        strings = str(ser.readline())
        if not strings:
            print_log("Received string is NULL", LVL_ERR)
            raise ValueError(-1)
        
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        index = strings.find('CSI_DATA')

        if index == -1:
            print_log("CSI_DATA field not found", LVL_ERR)
            raise ValueError(-2)

        csv_reader = csv.reader(StringIO(strings))
        csi_data = next(csv_reader)
        return csi_data
    
    else:

        print_log(f"get_csi_data - Reading CSV line of index {csv_index}", LVL_DBG)
            
        # csi_cvs_row = csv_array[csv_index % csv_len]
        csi_cvs_row = csv_array[csv_index]

        # Suppose (hardcoded) 25 fields
        header_arr = csi_cvs_row[0:23]
        data_arr = csi_cvs_row[23:]

        print_log(f"get_csi_data - Header fields: [{','.join(map(str, header_arr))}]", LVL_DBG)

        # Format data array as string
        data_str = "[" + ",".join(map(str, data_arr)) + "]"
        
        csi_data = ["CSI_DATA"] + header_arr + [data_str]

        print_log(f"get_csi_data - Final array len: {len(csi_data)}", LVL_DBG)
        
        csv_index += 1
        return csi_data
    

def fecth_csi_data_proc(port: str, from_ser: bool, hr_queue: deque):

    csi_data_source_init(port=port, from_ser=from_ser)

    while True:

        # get CSI data
        try:
            csi_data = get_csi_data()
        except ValueError as ve:
            if ve == 1:
                print_log("Error in get_csi_data", LVL_ERR)
            continue

        # Put into queue
        hr_queue.appendleft(csi_data)

        # Debug print
        print_log(f"Current CSI data queue length: {len(hr_queue)}", LVL_INF)
    
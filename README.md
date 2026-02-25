# WiFi-CSI-HeartRate
Group project for Industrial Application course, MSc in Computer Engineering, Università di Pisa, A.Y. 2025/2026.
Forked from the original repo [WiFi-CSI-HeartRate](https://github.com/AlessioSmn/WiFi-CSI-HeartRate).
Authors:
- [Carlo Mazzanti](https://github.com/cmsilen)
- [Alessio Simoncini](https://github.com/AlessioSmn)
## Description
System for the heart rate detection using CSI data of the WiFi subcarriers.
The purpose for each file is the following:
- arduino/hr_detect_and_send/hr_detect_and_send.ino code to be flashed in the board that will be connected to the MAX30102 sensor for ground truth. To be compiled using Arduino IDE
- esp/esp_csi_recv/main/main.c code of the CSI data receiver to be compiled with ESP-IDF
- esp/esp_csi_send/main/main.c code of the CSI data transmitter to be compiled with ESP-IDF
- raspberry/dbg_rx.py script for debugging what is received in the raspberry's serial line
- raspberry/features_raspberry.py functions for extracting features at runtime optimized for the raspberry board
- raspberry/hr_detection_lite_parallel.py script for starting the heart rate detection
- raspberry/parse_data.py functions for parsing the CSI data from the serial line optimized for the raspberry board
- raspberry/performance.py modified script for the heart rate detection that measures the computation time of each process
- stm32f4-discovery/Src/main.c code of the interface for the LCD screen
- workspace/check_data_balancing.py shows the balancing of the dataset
- workspace/collect_data.py script for building the dataset
- workspace/features_train_parallel.py extracts the features from the dataset and trains the model. It saves the model in workspace/models in .keras format
- workspace/to_tflite.py converts the model from the .keras format to the .tflite format

## Repository structure
There is one directory for each device used. You can find in workspace all script used on PC for the model workflow.
The dataset must be stored in workspace/data in csv format.

## Hardware
Raspberry Pi 3 Model B+

ESP-WROOM-32 (x3)

MAX30102 sensor

STM32F4-DISCOVERY

## Installation
### PC x86
Create a new virtual environment for training the LSTM model and converting it with TensorFlow Lite:
```bash
conda create -n tflite python=3.11
conda activate tflite
```
Install the following packages:
```bash
pip install pandas==2.3.3
pip install numpy==1.23.5
pip install tensorflow==2.12.0
pip install tflite-runtime==2.14.0
```

### Raspberry
Create a new virtual environment for training the LSTM model and converting it with TensorFlow Lite:
```bash
conda create -n tflite python=3.11
conda activate tflite
```
Install the following packages:
```bash
pip install pandas==3.0.0
pip install numpy==1.26.4
pip install pyserial==3.5
pip install scipy==1.17.0
pip install tflite-runtime==2.14.0
```

## Deployment
After the conversion of the model to .tflite, move the model from workspace/models to raspberry/models.
### Raspberry
```bash
python hr_detection_lite_parallel.py -p /dev/ttyUSB0 -ps /dev/ttyACM0
```

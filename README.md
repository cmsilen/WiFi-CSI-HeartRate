# WiFi-CSI-HeartRate
Group project for Industrial Application course, MSc in Computer Engineering, Università di Pisa, A.Y. 2025/2026
## Description
## Repository structure
## Hardware
Raspberry Pi 3 Model B+

ESP-WROOM-32 (x2)
## Deployment
### Raspberry
```bash
USR> python ./raspberry/main.py
```
Options available can be seen via:
```bash
USR> python ./raspberry/main.py -h

usage: main.py [-h] -p PORT [--csv] [-d {debug,info,error}]

Read CSI data from serial port, extract heart rate and display it graphically

options:
  -h, --help            show this help message and exit
  -p, --port PORT       Serial port of CSI device (or .csv filename, if --csv specified)
  --csv                 If specifies, it inticates to use the parameter -p as a .csv filename
  -d, --debug {debug,info,error}
                        Print level

    Examples:
     - from serial (COM3), debug level default (INFO)
        python main.py -p COM3

     - from serial (COM3), debug level DEBUG
        python main.py -p COM3 -d debug

     - from csv (file.csv), debug level default (INFO)
        python main.py -p file.csv --csv

     - from csv (file.csv), debug level DEBUG
        python main.py -p file.csv --csv -d debug

```

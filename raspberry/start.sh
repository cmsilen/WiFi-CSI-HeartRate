#!/bin/bash

# inizializza conda nella shell
source ~/miniconda3/etc/profile.d/conda.sh

# attiva l'ambiente
conda activate iaproject

# avvia python con priorità realtime
exec sudo chrt -f 50 python hr_detection.py -p /dev/ttyUSB0 -ps /dev/ttyACM0
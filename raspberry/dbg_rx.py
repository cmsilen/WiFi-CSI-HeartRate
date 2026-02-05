import serial
import time

# Sostituisci con la tua porta seriale
PORT = '/dev/ttyUSB0'  # /dev/tty.usbserial-0001
BAUD = 460800

# Apri la seriale
ser = serial.Serial(PORT, BAUD)

# Ora la scheda è resettata
print("ESP32 resettato")

time.sleep(0.1)  # breve pausa
ser.write(b"START\n")
ser.reset_input_buffer()

# Continua a leggere la seriale
while True:
    line = ser.readline()
    print(line.decode(errors='ignore').strip())

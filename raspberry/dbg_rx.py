import serial
import time

# Sostituisci con la tua porta seriale
PORT = '/dev/tty.usbserial-0001'  # su Linux /dev/ttyUSB0
BAUD = 921600

# Apri la seriale
ser = serial.Serial(PORT, BAUD)

# Induce reset via DTR/RTS
ser.dtr = False
ser.rts = True
time.sleep(0.1)  # breve pausa
ser.dtr = True
ser.rts = False

# Ora la scheda è resettata
print("ESP32 resettato")

time.sleep(0.1)  # breve pausa
ser.write(b"START\n")
ser.reset_input_buffer()

# Continua a leggere la seriale
while True:
    line = ser.readline()
    print(line.decode(errors='ignore').strip())

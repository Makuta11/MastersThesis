import serial
import time

# This creates the serial port to send information 
ser = serial.Serial('/dev/cu.usbmodem1101', 9600)

# Send information through port
ser.write(b'H') 

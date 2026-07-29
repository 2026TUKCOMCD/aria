import serial
import struct
import time

ser = serial.Serial('/dev/serial0', 115200, timeout=1)

def send_motor_command(mode, left_speed, right_speed):
    body = struct.pack('<HBBhh', 0xAA55, 2, mode, left_speed, right_speed)
    checksum = sum(body) & 0xFF
    tail = struct.pack('<H', 0x0A0D)
    packet = body + bytes([checksum]) + tail
    ser.write(packet)

print("360도 회전 시작")
start = time.time()
while time.time() - start < 10.6:
    send_motor_command(1, 100, -100)
    time.sleep(0.1)  # 0.1초마다 명령 전송 (타임아웃 1초보다 짧게)

send_motor_command(0, 0, 0)
print("정지")
ser.close()
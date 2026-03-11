import serial
import struct
from datetime import datetime
from typing import Optional


class PacketParser:
    """ESP32 바이너리 패킷을 파이썬 딕셔너리로 변환하는 클래스"""

    HEADER_BYTES = b"\x55\xAA"   # little-endian 0xAA55
    TAIL_BYTES = b"\x0D\x0A"     # little-endian 0x0A0D

    NAV_PACKET_SIZE = 38
    AIR_PACKET_SIZE = 22
    ODOM_PACKET_SIZE = 34

    def __init__(self, port: str = "/dev/serial0", baudrate: int = 115200):
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
            print(f"시리얼 포트 연결 성공: {port}")
        except Exception as e:
            print(f"시리얼 포트 열기 실패: {e}")
            raise

    def calculate_checksum(self, data: bytes) -> int:
        """패킷의 마지막 3바이트(checksum 1 + tail 2)를 제외한 합"""
        return sum(data[:-3]) & 0xFF

    def parse_nav_packet(self, data: bytes) -> Optional[dict]:
        try:
            unpacked = struct.unpack("<H B f f f f f f f f B H", data)

            # unpacked index
            # 0 header, 1 id, 2 tof1, 3 tof2, 4 accx, 5 accy, 6 accz,
            # 7 gyrox, 8 gyroy, 9 gyroz, 10 checksum, 11 tail
            if unpacked[10] != self.calculate_checksum(data):
                return None

            return {
                "type": "NAV",
                "tof1": unpacked[2],
                "tof2": unpacked[3],
                "acc": {"x": unpacked[4], "y": unpacked[5], "z": unpacked[6]},
                "gyro": {"x": unpacked[7], "y": unpacked[8], "z": unpacked[9]},
                "timestamp": datetime.now(),
            }
        except Exception:
            return None

    def parse_air_packet(self, data: bytes) -> Optional[dict]:
        try:
            unpacked = struct.unpack("<H B f f f f B H", data)

            # 0 header, 1 id, 2 pm25, 3 voc, 4 temp, 5 humi, 6 checksum, 7 tail
            if unpacked[6] != self.calculate_checksum(data):
                return None

            return {
                "type": "AIR",
                "pm25": unpacked[2],
                "voc": unpacked[3],
                "temp": unpacked[4],
                "humi": unpacked[5],
                "timestamp": datetime.now(),
            }
        except Exception:
            return None

    def parse_odom_packet(self, data: bytes) -> Optional[dict]:
        try:
            unpacked = struct.unpack("<H B i i f f f f f B H", data)

            # 0 header, 1 id, 2 encL, 3 encR, 4 x, 5 y, 6 theta, 7 lin, 8 ang, 9 checksum, 10 tail
         #   if unpacked[9] != self.calculate_checksum(data):
          #      return None

            return {
                "type": "ODOM",
                "enc": {"l": unpacked[2], "r": unpacked[3]},
                "pos": {"x": unpacked[4], "y": unpacked[5], "theta": unpacked[6]},
                "vel": {"lin": unpacked[7], "ang": unpacked[8]},
                "timestamp": datetime.now(),
            }
        except Exception:
            return None

    def read_packet(self) -> Optional[dict]:
        """시리얼 버퍼를 감시하다가 유효한 패킷 1개를 파싱해 반환"""
        while self.ser.in_waiting > 0:
            b1 = self.ser.read(1)

            if b1 != b"\x55":
                continue

            b2 = self.ser.read(1)
            if b2 != b"\xAA":
                continue

            p_id_b = self.ser.read(1)
            if not p_id_b:
                return None

            p_id = p_id_b[0]

            if p_id == 0:  # NAV
                rest = self.ser.read(35)
                data = b"\x55\xAA" + p_id_b + rest
                if len(data) == self.NAV_PACKET_SIZE:
                    return self.parse_nav_packet(data)

            elif p_id == 1:  # AIR
                rest = self.ser.read(19)
                data = b"\x55\xAA" + p_id_b + rest
                if len(data) == self.AIR_PACKET_SIZE:
                    return self.parse_air_packet(data)

            elif p_id == 3:  # ODOM
                rest = self.ser.read(31)
                data = b"\x55\xAA" + p_id_b + rest
                if len(data) == self.ODOM_PACKET_SIZE:
                    return self.parse_odom_packet(data)

        return None

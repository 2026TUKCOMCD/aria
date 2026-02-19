import serial
import struct
from datetime import datetime
from typing import Optional

class PacketParser:
    """ESP32 바이너리 패킷을 파이썬 딕셔너리로 변환하는 클래스"""
    HEADER = 0xAA55
    TAIL = 0x0A0D

    # ESP32 구조체 크기와 일치해야 함
    NAV_PACKET_SIZE = 38
    AIR_PACKET_SIZE = 22
    ODOM_PACKET_SIZE = 34

    def __init__(self, port='/dev/serial0', baudrate=115200):
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
            print(f"시리얼 포트 연결 성공: {port}")
        except Exception as e:
            print(f"시리얼 포트 열기 실패: {e}")
            raise

    def calculate_checksum(self, data: bytes) -> int:
        """패킷의 데이터 합계를 계산하여 체크섬 검증 (마지막 3바이트 제외)"""
        return sum(data[:-3]) & 0xFF

    def parse_nav_packet(self, data: bytes) -> Optional[dict]:
        try:
            # < : Little Endian / H: uint16, B: uint8, f: float
            unpacked = struct.unpack('<H B f f f f f f f f B H', data)
            if unpacked[10] != self.calculate_checksum(data): return None
            return {
                'type': 'NAV',
                'tof1': unpacked[2], 'tof2': unpacked[3],
                'acc': {'x': unpacked[4], 'y': unpacked[5], 'z': unpacked[6]},
                'gyro': {'x': unpacked[7], 'y': unpacked[8], 'z': unpacked[9]},
                'timestamp': datetime.now()
            }
        except Exception: return None

    def parse_air_packet(self, data: bytes) -> Optional[dict]:
        try:
            # AirPacket: Header(2), ID(1), PM2.5(4), VOC(4), Temp(4), Humi(4), Checksum(1), Tail(2)
            unpacked = struct.unpack('<H B f f f f B H', data)
            if unpacked[6] != self.calculate_checksum(data): return None
            return {
                'type': 'AIR',
                'pm25': unpacked[2], 'voc': unpacked[3],
                'temp': unpacked[4], 'humi': unpacked[5],
                'timestamp': datetime.now()
            }
        except Exception: return None

    def parse_odom_packet(self, data: bytes) -> Optional[dict]:
        try:
            # OdomPacket: Header(2), ID(1), EncL(4), EncR(4), PosX(4), PosY(4), Theta(4), LinV(4), AngV(4), Checksum(1), Tail(2)
            unpacked = struct.unpack('<H B i i f f f f f B H', data)
            if unpacked[9] != self.calculate_checksum(data): return None
            return {
                'type': 'ODOM',
                'enc': {'l': unpacked[2], 'r': unpacked[3]},
                'pos': {'x': unpacked[4], 'y': unpacked[5], 'theta': unpacked[6]},
                'vel': {'lin': unpacked[7], 'ang': unpacked[8]},
                'timestamp': datetime.now()
            }
        except Exception: return None

    def read_packet(self) -> Optional[dict]:
        """시리얼 버퍼를 감시하다가 유효한 패킷이 들어오면 파싱 결과를 반환"""
        if self.ser.in_waiting > 0:
            b = self.ser.read(1)
            # 0x55 0xAA 순서로 들어오는지 확인 (Little Endian 기준)
            if b == b'\x55':
                if self.ser.read(1) == b'\xAA':
                    p_id_b = self.ser.read(1)
                    if not p_id_b: return None
                    p_id = p_id_b[0]

                    # ID에 따라 나머지 데이터 읽기
                    if p_id == 0: # NAV
                        return self.parse_nav_packet(b'\x55\xAA' + p_id_b + self.ser.read(35))
                    elif p_id == 1: # AIR
                        return self.parse_air_packet(b'\x55\xAA' + p_id_b + self.ser.read(19))
                    elif p_id == 3: # ODOM
                        return self.parse_odom_packet(b'\x55\xAA' + p_id_b + self.ser.read(31))
        return None
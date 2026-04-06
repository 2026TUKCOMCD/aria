import serial
import struct
import time
from datetime import datetime
from typing import Optional


class PacketParser:
    """ESP32 바이너리 패킷을 파이썬 딕셔너리로 변환하고, 모터 명령을 송신하는 클래스"""

    HEADER = 0xAA55
    TAIL   = 0x0A0D

    # 수신 패킷 크기 (ESP32 구조체 크기와 일치)
    NAV_PACKET_SIZE  = 38
    AIR_PACKET_SIZE  = 22
    ODOM_PACKET_SIZE = 34

    # MotorCommandPacket 크기: Header(2)+ID(1)+Mode(1)+LeftSpeed(2)+RightSpeed(2)+Checksum(1)+Tail(2) = 11
    MOTOR_CMD_PACKET_SIZE = 11

    # DriveMode (ESP32 펌웨어와 동일)
    MODE_STOP   = 0
    MODE_MANUAL = 1
    MODE_AUTO   = 2

    def __init__(self, port='/dev/serial0', baudrate=115200):
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
            print(f"시리얼 포트 연결 성공: {port}")
        except Exception as e:
            print(f"시리얼 포트 열기 실패: {e}")
            raise

    # ──────────────────────────────────────────────
    # 체크섬
    # ──────────────────────────────────────────────

    def calculate_checksum(self, data: bytes) -> int:
        """패킷 체크섬 계산 (마지막 3바이트 제외한 합산, 하위 8비트)"""
        return sum(data[:-3]) & 0xFF

    # ──────────────────────────────────────────────
    # 수신: 패킷 파싱
    # ──────────────────────────────────────────────

    def parse_nav_packet(self, data: bytes) -> Optional[dict]:
        try:
            unpacked = struct.unpack('<H B f f f f f f f f B H', data)
            if unpacked[10] != self.calculate_checksum(data):
                return None
            return {
                'type': 'NAV',
                'tof1': unpacked[2], 'tof2': unpacked[3],
                'acc':  {'x': unpacked[4], 'y': unpacked[5], 'z': unpacked[6]},
                'gyro': {'x': unpacked[7], 'y': unpacked[8], 'z': unpacked[9]},
                'timestamp': datetime.now()
            }
        except Exception:
            return None

    def parse_air_packet(self, data: bytes) -> Optional[dict]:
        try:
            unpacked = struct.unpack('<H B f f f f B H', data)
            if unpacked[6] != self.calculate_checksum(data):
                return None
            return {
                'type': 'AIR',
                'pm25': unpacked[2], 'voc': unpacked[3],
                'temp': unpacked[4], 'humi': unpacked[5],
                'timestamp': datetime.now()
            }
        except Exception:
            return None

    def parse_odom_packet(self, data: bytes) -> Optional[dict]:
        try:
            unpacked = struct.unpack('<H B i i f f f f f B H', data)
            if unpacked[9] != self.calculate_checksum(data):
                return None
            return {
                'type': 'ODOM',
                'enc': {'l': unpacked[2], 'r': unpacked[3]},
                'pos': {'x': unpacked[4], 'y': unpacked[5], 'theta': unpacked[6]},
                'vel': {'lin': unpacked[7], 'ang': unpacked[8]},
                'timestamp': datetime.now()
            }
        except Exception:
            return None

    def read_packet(self) -> Optional[dict]:
        """수신 버퍼에서 패킷을 하나 읽어 딕셔너리로 반환"""
        while self.ser.in_waiting > 0:
            b = self.ser.read(1)
            if b != b'\x55':
                continue

            b2 = self.ser.read(1)
            if b2 != b'\xAA':
                continue

            p_id_b = self.ser.read(1)
            if not p_id_b:
                return None
            p_id = p_id_b[0]

            if p_id == 0:    # NAV
                data = b'\x55\xAA' + p_id_b + self.ser.read(35)
                if len(data) == self.NAV_PACKET_SIZE:
                    return self.parse_nav_packet(data)

            elif p_id == 1:  # AIR
                data = b'\x55\xAA' + p_id_b + self.ser.read(19)
                if len(data) == self.AIR_PACKET_SIZE:
                    return self.parse_air_packet(data)

            elif p_id == 3:  # ODOM
                data = b'\x55\xAA' + p_id_b + self.ser.read(31)
                if len(data) == self.ODOM_PACKET_SIZE:
                    return self.parse_odom_packet(data)

        return None

    # ──────────────────────────────────────────────
    # 송신: 모터 명령
    # ──────────────────────────────────────────────

    def send_motor_command(self, left_speed: int, right_speed: int,
                           mode: int = MODE_MANUAL) -> bool:
        """
        ESP32로 모터 명령 패킷을 전송한다.

        MotorCommandPacket 구조 (리틀엔디언):
          Header   : 0x55 0xAA  (uint16, 2바이트)
          ID       : 0x02       (uint8,  1바이트)
          Mode     : 0/1/2      (uint8,  1바이트)
          LeftSpeed: -255~255   (int16,  2바이트)
          RightSpeed:-255~255   (int16,  2바이트)
          Checksum :            (uint8,  1바이트)
          Tail     : 0x0A 0x0D  (uint16, 2바이트)
        총 11바이트
        """
        left_speed  = max(-255, min(255, left_speed))
        right_speed = max(-255, min(255, right_speed))

        packet = struct.pack('<H B B h h B H',
                             0xAA55,
                             0x02,
                             mode,
                             left_speed,
                             right_speed,
                             0x00,
                             0x0A0D)

        checksum = self.calculate_checksum(packet)
        packet = packet[:8] + bytes([checksum]) + packet[9:]

        try:
            self.ser.write(packet)
            return True
        except Exception as e:
            print(f"[PacketParser] 모터 명령 전송 실패: {e}")
            return False

    def send_stop(self) -> bool:
        """모터 정지 명령 전송 (MODE_STOP)"""
        return self.send_motor_command(0, 0, mode=self.MODE_STOP)

    # ──────────────────────────────────────────────
    # 송신: 360도 회전
    # ──────────────────────────────────────────────

    def send_rotate_360(self, speed: int = 50, clockwise: bool = True,
                        duration_sec: float = 22.0) -> bool:
        """
        시간 기반으로 360도 회전.
        MOTOR_TIMEOUT=30000 환경에서 단순하고 안정적.

        Args:
            speed:        회전 속도 (0~255, 기본 150)
            clockwise:    True=시계 방향, False=반시계 방향
            duration_sec: 회전 시간(초), speed=150 기준 360도 = 약 10.6초
        """
        speed = max(0, min(255, speed))

        if clockwise:
            left_speed  =  speed
            right_speed = -speed
        else:
            left_speed  = -speed
            right_speed =  speed

        print(f"[PacketParser] 360도 회전 시작 "
              f"({'시계' if clockwise else '반시계'} 방향, speed={speed}, {duration_sec}초)")

        self.send_motor_command(left_speed, right_speed, mode=self.MODE_MANUAL)
        time.sleep(duration_sec)
        self.send_stop()

        print("[PacketParser] 360도 회전 완료 → 정지")
        return True

import serial
import struct
import time
from datetime import datetime
from typing import Optional

class PacketParser:
    HEADER = 0xAA55
    TAIL   = 0x0A0D

    NAV_PACKET_SIZE  = 42
    AIR_PACKET_SIZE  = 22
    ODOM_PACKET_SIZE = 34
    MOTOR_CMD_PACKET_SIZE = 11

    MODE_STOP   = 0
    MODE_MANUAL = 1
    MODE_AUTO   = 2

    def __init__(self, port='/dev/serial0', baudrate=115200):
        try:
            # 타임아웃 없이 논블로킹으로 설정
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0)
            self.buffer = bytearray()
            print(f"시리얼 포트 연결 성공: {port}")
        except Exception as e:
            print(f"시리얼 포트 열기 실패: {e}")
            raise

    def calculate_checksum(self, data: bytes) -> int:
        return sum(data[:-3]) & 0xFF

    def parse_odom_packet(self, data: bytes) -> Optional[dict]:
        try:
            unpacked = struct.unpack('<H B i i f f f f f B H', data)
            calc_chk = self.calculate_checksum(data)
            
            # 꼬리표 검사는 생략하고 강력한 체크섬만 검사합니다.
            if unpacked[9] != calc_chk:
                print(f"⚠️ [파서 에러] ODOM 체크섬 불일치! 수신값:{unpacked[9]}, 계산값:{calc_chk}")
                return None
                
            return {
                'type': 'ODOM',
                'enc': {'l': unpacked[2], 'r': unpacked[3]},
                # 💡 [핵심 수정] 노드에서 바로 읽을 수 있도록 posX, posY, posTheta로 빼줍니다.
                'posX': unpacked[4],
                'posY': unpacked[5],
                'posTheta': unpacked[6],
                'vel': {'lin': unpacked[7], 'ang': unpacked[8]},
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"⚠️ [파서 에러] ODOM 파싱 실패: {e}")
            return None

    def parse_nav_packet(self, data: bytes) -> Optional[dict]:
        try:
            unpacked = struct.unpack('<H B f f f f f f f f f B H', data)
            if unpacked[11] != self.calculate_checksum(data): return None
            return {
                'type': 'NAV',
                'tof1': unpacked[2], 'tof2': unpacked[3], 'tof3': unpacked[4],
                'acc':  {'x': unpacked[5], 'y': unpacked[6], 'z': unpacked[7]},
                'gyro': {'x': unpacked[8], 'y': unpacked[9], 'z': unpacked[10]},
                'timestamp': datetime.now()
            }
        except Exception:
            return None

    def parse_air_packet(self, data: bytes) -> Optional[dict]:
        try:
            unpacked = struct.unpack('<H B f f f f B H', data)
            if unpacked[6] != self.calculate_checksum(data): return None
            return {
                'type': 'AIR',
                'pm25': unpacked[2], 'voc': unpacked[3],
                'temp': unpacked[4], 'humi': unpacked[5],
                'timestamp': datetime.now()
            }
        except Exception:
            return None

    def read_packet(self) -> Optional[dict]:
        # 1. 버퍼에 쌓인 데이터를 모조리 가져옵니다.
        if self.ser.in_waiting > 0:
            self.buffer.extend(self.ser.read(self.ser.in_waiting))

        # 2. 헤더(0x55, 0xAA)를 찾아 패킷을 조립합니다.
        while len(self.buffer) >= 3:
            if self.buffer[0] != 0x55 or self.buffer[1] != 0xAA:
                self.buffer.pop(0)
                continue

            p_id = self.buffer[2]
            if p_id == 0: expected_len = self.NAV_PACKET_SIZE
            elif p_id == 1: expected_len = self.AIR_PACKET_SIZE
            elif p_id == 3: expected_len = self.ODOM_PACKET_SIZE
            else:
                self.buffer.pop(0)
                continue

            # 버퍼에 아직 전체 패킷이 안 들어왔다면 다음번을 기약합니다.
            if len(self.buffer) < expected_len:
                return None

            # 완전한 패킷이 모였으므로 잘라냅니다.
            packet_data = bytes(self.buffer[:expected_len])
            del self.buffer[:expected_len]

            # 파싱 진행
            if p_id == 0: return self.parse_nav_packet(packet_data)
            elif p_id == 1: return self.parse_air_packet(packet_data)
            elif p_id == 3: return self.parse_odom_packet(packet_data)

        return None

    def send_motor_command(self, left_speed: int, right_speed: int,
                           mode: int = MODE_MANUAL) -> bool:
        left_speed  = max(-255, min(255, left_speed))
        right_speed = max(-255, min(255, right_speed))
        data_to_sum = struct.pack('<H B B h h', 0xAA55, 0x02, mode, left_speed, right_speed)
        checksum = sum(data_to_sum) & 0xFF
        packet = data_to_sum + struct.pack('<B H', checksum, 0x0A0D)
        try:
            self.ser.write(packet)
            return True
        except Exception as e:
            print(f"[PacketParser] 모터 명령 전송 실패: {e}")
            return False

    def send_stop(self) -> bool:
        return self.send_motor_command(0, 0, mode=self.MODE_STOP)

    def send_fan_command(self, fan_speed: int) -> bool:
        # 팬 속도를 0 ~ 255 사이로 안전하게 제한
        fan_speed = max(0, min(255, fan_speed))
        
        # 패킷 조립: <H(헤더 2바이트) B(ID 1바이트) B(속도 1바이트)
        # ID는 ESP32 코드와 동일하게 4 (0x04)를 사용합니다.
        data_to_sum = struct.pack('<H B B', 0xAA55, 0x04, fan_speed)
        
        # 체크섬 계산 및 꼬리표 붙이기
        checksum = sum(data_to_sum) & 0xFF
        packet = data_to_sum + struct.pack('<B H', checksum, 0x0A0D)
        
        try:
            self.ser.write(packet)
            return True
        except Exception as e:
            print(f"[PacketParser] 팬 명령 전송 실패: {e}")
            return False

    def send_rotate_360(self, speed: int = 50, clockwise: bool = True,
                        duration_sec: float = 22.0) -> bool:
        speed = max(0, min(255, speed))
        left_speed, right_speed = (speed, -speed) if clockwise else (-speed, speed)
        print(f"[PacketParser] 360도 회전 시작 (speed={speed})")
        self.send_motor_command(left_speed, right_speed, mode=self.MODE_MANUAL)
        time.sleep(duration_sec)
        self.send_stop()
        return True

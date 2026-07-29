"""
ESP32 통신 모듈

UART 기반 패킷 통신

Author: 박진주
"""

import serial
import struct
import time


class ESP32Controller:
    """
    ESP32 통신 클래스
    
    Features:
    - 모터 제어 (MotorCommandPacket)
    - 센서 데이터 수신 (NavPacket, AirPacket, OdomPacket)
    """
    
    def __init__(self, port='/dev/ttyAMA0', baudrate=115200, verbose=True):
        """
        Args:
            port (str): UART 포트
            baudrate (int): 통신 속도
            verbose (bool): 로그 출력
        """
        self.port = port
        self.baudrate = baudrate
        self.verbose = verbose
        
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1
            )
            
            if self.verbose:
                print(f"[ESP32] Connected to {self.port}")
        
        except Exception as e:
            if self.verbose:
                print(f"[ESP32] ❌ Connection failed: {e}")
            self.serial = None
    
    def _calculate_checksum(self, data):
        """체크섬 계산"""
        return sum(data[:-3]) & 0xFF
    
    def send_motor_command(self, mode, left_speed, right_speed):
        """
        모터 제어 패킷 전송
        
        Args:
            mode (int): 0=정지, 1=수동, 2=자동
            left_speed (int): -255 ~ 255
            right_speed (int): -255 ~ 255
            
        Returns:
            bool: 성공 여부
        """
        if self.serial is None:
            return False
        
        # 속도 제한
        left_speed = max(-255, min(255, left_speed))
        right_speed = max(-255, min(255, right_speed))
        
        # 패킷 생성
        packet = bytearray()
        packet += struct.pack('<H', 0xAA55)  # header
        packet += struct.pack('B', 2)        # id
        packet += struct.pack('B', mode)     # mode
        packet += struct.pack('<h', left_speed)   # leftSpeed
        packet += struct.pack('<h', right_speed)  # rightSpeed
        
        # 체크섬
        checksum = self._calculate_checksum(packet)
        packet += struct.pack('B', checksum)
        
        # tail
        packet += struct.pack('<H', 0x0A0D)
        
        try:
            self.serial.write(packet)
            
            if self.verbose:
                print(f"[ESP32] Motor: mode={mode}, L={left_speed}, R={right_speed}")
            
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"[ESP32] ❌ Send failed: {e}")
            return False
    
    def stop(self):
        """긴급 정지"""
        return self.send_motor_command(mode=0, left_speed=0, right_speed=0)
    
    def move_forward(self, speed=100):
        """전진"""
        return self.send_motor_command(mode=1, left_speed=speed, right_speed=speed)
    
    def move_backward(self, speed=100):
        """후진"""
        return self.send_motor_command(mode=1, left_speed=-speed, right_speed=-speed)
    
    def turn_left(self, speed=100):
        """좌회전 (제자리)"""
        return self.send_motor_command(mode=1, left_speed=-speed, right_speed=speed)
    
    def turn_right(self, speed=100):
        """우회전 (제자리)"""
        return self.send_motor_command(mode=1, left_speed=speed, right_speed=-speed)
    
    def rotate_360(self, speed=80, duration=10.0):
        """
        360도 회전
        
        Args:
            speed (int): 회전 속도
            duration (float): 회전 시간 (초)
        """
        if self.verbose:
            print(f"[ESP32] Rotating 360° ({duration}s)")
        
        # 회전 시작
        self.turn_right(speed)
        
        # 대기
        time.sleep(duration)
        
        # 정지
        self.stop()
        
        if self.verbose:
            print(f"[ESP32] ✅ Rotation complete")
    
    def cleanup(self):
        """종료 처리"""
        if self.serial is not None:
            self.stop()
            self.serial.close()
            
            if self.verbose:
                print("[ESP32] Disconnected")


# ==================== 테스트 ====================
if __name__ == '__main__':
    print("="*60)
    print("ESP32 Controller Test")
    print("="*60)
    
    esp = ESP32Controller(port='/dev/ttyAMA0')
    
    try:
        # 테스트 1: 전진
        print("\n[Test 1] Forward")
        esp.move_forward(100)
        time.sleep(2)
        esp.stop()
        
        # 테스트 2: 360도 회전
        print("\n[Test 2] 360° Rotation")
        esp.rotate_360(speed=80, duration=10.0)
        
        print("\n✅ Test complete")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        esp.cleanup()
"""
팬 동작 확인 스크립트
ESP32로부터 AirPacket 수신 → PM2.5 값 + 현재 팬 속도 출력
"""
import serial
import struct
import time

# ── 설정 ──────────────────────────────────────────────────────────────
SERIAL_PORT = '/dev/ttyS0'   # 라즈베리파이 UART 포트
BAUD_RATE   = 115200
# ─────────────────────────────────────────────────────────────────────

# AirPacket: header(2) + id(1) + pm25(4) + voc(4) + temp(4) + humi(4) + checksum(1) + tail(2) = 22 bytes
AIR_PACKET_ID   = 1
AIR_PACKET_SIZE = 22


def fan_speed_from_pm25(pm25: float) -> tuple:
    """PM2.5 → (퍼센트, PWM값, 설명)"""
    if pm25 < 0:
        return 50, 128, "PM 데이터 없음 (기본 50%)"
    elif pm25 < 15:
        return 30, 77,  f"낮음 ({pm25:.1f} μg/m³)"
    elif pm25 < 35:
        return 50, 128, f"보통 ({pm25:.1f} μg/m³)"
    elif pm25 < 75:
        return 80, 204, f"높음 ({pm25:.1f} μg/m³)"
    else:
        return 100, 255, f"매우 높음 ({pm25:.1f} μg/m³)"


def parse_air_packet(data: bytes):
    """AirPacket 파싱 → (pm25, voc, temp, humi) or None"""
    if len(data) < AIR_PACKET_SIZE:
        return None
    # header: 0x55 0xAA  (little-endian)
    if data[0] != 0x55 or data[1] != 0xAA:
        return None
    if data[2] != AIR_PACKET_ID:
        return None

    pm25, voc, temp, humi = struct.unpack_from('<ffff', data, 3)
    return pm25, voc, temp, humi


def main():
    print(f"[연결] {SERIAL_PORT} @ {BAUD_RATE} baud")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    except serial.SerialException as e:
        print(f"[오류] 시리얼 포트 열기 실패: {e}")
        return

    print("AirPacket 대기 중... (Ctrl+C로 종료)\n")
    buf = bytearray()

    try:
        while True:
            buf.extend(ser.read(64))

            # 헤더(0x55 0xAA) 위치 탐색
            while len(buf) >= AIR_PACKET_SIZE:
                idx = -1
                for i in range(len(buf) - 1):
                    if buf[i] == 0x55 and buf[i + 1] == 0xAA:
                        idx = i
                        break

                if idx == -1:
                    buf = buf[-1:]
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < AIR_PACKET_SIZE:
                    break

                result = parse_air_packet(bytes(buf[:AIR_PACKET_SIZE]))
                buf = buf[AIR_PACKET_SIZE:]

                if result is None:
                    continue

                pm25, voc, temp, humi = result
                pct, pwm, desc = fan_speed_from_pm25(pm25)

                print("=" * 40)
                print(f"  PM2.5    : {pm25:.1f} μg/m³")
                print(f"  VOC      : {voc:.0f}")
                print(f"  온도/습도 : {temp:.1f}°C / {humi:.1f}%")
                print(f"  팬 속도  : {pct}%  (PWM {pwm}/255)  {desc}")
                if pct > 0:
                    print(f"  → 팬이 돌아야 합니다 ✓")
                print("=" * 40)

    except KeyboardInterrupt:
        print("\n종료")
    finally:
        ser.close()


if __name__ == '__main__':
    main()

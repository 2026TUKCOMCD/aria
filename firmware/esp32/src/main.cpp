/*
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * ARIA - 구동 + 공기질 통합 펌웨어
 * 기준:
 *  - 현재 정상 동작 중인 모터/엔코더/UART/ToF/IMU 핀 유지
 *  - AirPacket은 예전 라즈베리파이 파서와 호환되게 id=1, 22 bytes 유지
 *  - NavPacket은 ToF 3개 포함, 42 bytes
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 */

#include <Arduino.h>
#include <Wire.h>
#include "Adafruit_VL53L0X.h"
#include "MPU9250.h"
#include "CytronMotorDriver.h"
#include "Adafruit_SGP40.h"
#include "DHT.h"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [1] 핀 설정
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// I2C: ToF, IMU, SGP40 공용
#define I2C_SDA 8
#define I2C_SCL 9

// ToF XSHUT
#define TOF1_XSHUT 4 //좌
#define TOF2_XSHUT 5 //우
#define TOF3_XSHUT 6 //중앙

// 왼쪽 모터
#define MOTOR1_PWM 10
#define MOTOR1_DIR 11

// 오른쪽 모터
#define MOTOR2_PWM 12
#define MOTOR2_DIR 13

// 왼쪽 엔코더
#define ENCODER1_A 14
#define ENCODER1_B 15

// 오른쪽 엔코더
#define ENCODER2_A 16
#define ENCODER2_B 17

// Raspberry Pi 통신 UART
#define TX2_PIN 18
#define RX2_PIN 21

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [1-2] 공기질 센서 / 팬 핀
// 기존 핀과 겹치지 않게 새로 배정
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#define DHT_PIN       7
#define DHT_TYPE      DHT22

#define PM_RX_PIN     35   // PM센서 TX -> ESP32 RX
#define PM_TX_PIN     36   // ESP32 TX -> PM센서 RX, 필요 없으면 미연결 가능

#define FAN_PIN       38   // MOSFET Gate
#define FAN_PWM_CH    2
#define FAN_PWM_FREQ  25000
#define FAN_PWM_RES   8

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [2] 로봇 파라미터
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#define ENCODER_PPR    16
#define GEAR_RATIO     131.0
#define WHEEL_DIAMETER 0.096
#define WHEEL_BASE     0.23

#define RAMP_STEP      5
#define RAMP_INTERVAL  20
#define MOTOR_TIMEOUT  10000

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [3] 통신 패킷
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma pack(push, 1)

struct NavPacket {
    uint16_t header = 0xAA55;
    uint8_t  id     = 0;

    float tof1Distance;
    float tof2Distance;
    float tof3Distance;

    float accelX;
    float accelY;
    float accelZ;

    float gyroX;
    float gyroY;
    float gyroZ;

    uint8_t  checksum;
    uint16_t tail = 0x0A0D;
};

struct FanCommandPacket {
    uint16_t header = 0xAA55;
    uint8_t  id     = 4; // 팬 전용 ID는 4번으로 부여

    uint8_t  fanSpeed;   // 0 ~ 255

    uint8_t  checksum;
    uint16_t tail = 0x0A0D;
};

struct AirPacket {
    uint16_t header = 0xAA55;
    uint8_t  id     = 1;

    float pm25;
    float voc;
    float temperature;
    float humidity;

    uint8_t  checksum;
    uint16_t tail = 0x0A0D;
};

struct OdomPacket {
    uint16_t header = 0xAA55;
    uint8_t  id     = 3;

    int32_t leftEncoderCount;
    int32_t rightEncoderCount;

    float posX;
    float posY;
    float posTheta;

    float linearVel;
    float angularVel;

    uint8_t  checksum;
    uint16_t tail = 0x0A0D;
};

struct MotorCommandPacket {
    uint16_t header = 0xAA55;
    uint8_t  id     = 2;
    uint8_t  mode;

    int16_t leftSpeed;
    int16_t rightSpeed;

    uint8_t  checksum;
    uint16_t tail = 0x0A0D;
};

#pragma pack(pop)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [4] 전역 객체 및 변수
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Adafruit_VL53L0X tof1, tof2, tof3;

MPU9250 imu(Wire, 0x68);

CytronMD motor1(PWM_DIR, MOTOR1_PWM, MOTOR1_DIR);
CytronMD motor2(PWM_DIR, MOTOR2_PWM, MOTOR2_DIR);

// -------------------------------------------------------------
// [수정] 오프셋 상수 및 바퀴축 전용 누적 변수 추가
// -------------------------------------------------------------
const float X_OFFSET = 0.02f;  // ★본인이 측정한 미터 단위 수치로 변경하세요 (예: 4cm = 0.04f)

float w_odomX = 0.0f;          // 바퀴축 중심 X (내부 누적용)
float w_odomY = 0.0f;          // 바퀴축 중심 Y (내부 누적용)
// -------------------------------------------------------------

// 최신 IMU 데이터를 상시 보관할 전역 변수 바구니 선언
float latest_accelX = 0.0f; float latest_accelY = 0.0f; float latest_accelZ = 0.0f;
float latest_gyroX  = 0.0f; float latest_gyroY  = 0.0f; float latest_gyroZ  = 0.0f;
unsigned long lastImuRead = 0; // IMU 개별 수집 주기용 타이머

#define MOTOR1_SIGN 1
#define MOTOR2_SIGN 1

int16_t toPhysicalMotor1Speed(int16_t logicalSpeed) {
    return constrain((int16_t)(logicalSpeed * MOTOR1_SIGN), -255, 255);
}

int16_t toPhysicalMotor2Speed(int16_t logicalSpeed) {
    return constrain((int16_t)(logicalSpeed * MOTOR2_SIGN), -255, 255);
}

// 공기질
Adafruit_SGP40 sgp40;
DHT dht(DHT_PIN, DHT_TYPE);
HardwareSerial PMSerial(1);

struct MotorState {
    int16_t targetSpeed  = 0;
    int16_t currentSpeed = 0;
    unsigned long lastUpdate = 0;
};

enum DriveMode {
    MODE_STOP = 0,
    MODE_MANUAL = 1,
    MODE_AUTO = 2
};

MotorState motor1State;
MotorState motor2State;
DriveMode currentMode = MODE_STOP;

volatile int32_t encoder1Count = 0;
volatile int32_t encoder2Count = 0;

int32_t lastEncoder1Count = 0;
int32_t lastEncoder2Count = 0;

float odomX = 0.0f;
float odomY = 0.0f;
float odomTheta = 0.0f;

float linearVelocity = 0.0f;
float angularVelocity = 0.0f;

unsigned long lastOdomUpdate   = 0;
unsigned long lastNav          = 0;
unsigned long lastOdom         = 0;
unsigned long lastAir          = 0;
unsigned long lastMotorCommand = 0;
unsigned long lastFanUpdate    = 0;

bool imuReady  = false;
bool tof1Ready = false;
bool tof2Ready = false;
bool tof3Ready = false;

bool sgp40Ready = false;
bool dhtReady   = false;
bool pmReady    = false;

// PM 센서 데이터
struct PMSensorData {
    uint16_t pm1_0 = 0;
    uint16_t pm2_5 = 0;
    uint16_t pm10  = 0;
    bool dataValid = false;
};

PMSensorData pmData;
uint8_t pmBuffer[32];
uint8_t pmBufferIndex = 0;

// 공기질 값
float airTemperature = -1.0f;
float airHumidity    = -1.0f;
float vocValue       = -1.0f;

// 팬 제어
uint8_t fanSpeed = 0;
uint8_t targetFanSpeed = 0;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [5] 엔코더 인터럽트
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void IRAM_ATTR encoder1_ISR() {
    // 왼쪽 바퀴 전진 시 증가
    encoder1Count += (digitalRead(ENCODER1_B) > 0) ? 1 : -1;
}

void IRAM_ATTR encoder2_ISR() {
    // 오른쪽 바퀴 전진 시 증가
    encoder2Count += (digitalRead(ENCODER2_B) > 0) ? -1 : 1;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [6] 유틸리티
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

uint8_t calculateChecksum(uint8_t* data, size_t len) {
    uint8_t sum = 0;
    for (size_t i = 0; i < len - 3; i++) {
        sum += data[i];
    }
    return sum;
}

void i2cScan() {
    Serial.println("\n[I2C 스캔]");
    int found = 0;

    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            Serial.printf("  ✓ 0x%02X\n", addr);
            found++;
        }
    }

    Serial.printf("  총 %d개\n\n", found);
}

// VL53L0X 거리 읽기
float readTof0xDistance(Adafruit_VL53L0X &tof) {
    if (!tof.isRangeComplete()) {
        return -1.0f;
    }

    uint16_t d = tof.readRangeResult();
    // ⚠️ 핵심: 호환 센서 특유의 에러 코드(8190, 8191) 및 타임아웃(65535) 제거
    if (d == 0xFFFF || d >= 8190) {
        return -1.0f;
    }
    return (d == 0xFFFF) ? -1.0f : (float)d;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [7] PM 센서 UART 처리
// PMS5003 / PMS7003 계열 32바이트 프레임 기준
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ★ [ToF3 사양 변경점] 기존 VL53L1X 전용 거리 측정 함수(readTof1xDistance)는 완전히 삭제했습니다.

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [7] PM 센서 UART 처리
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool readPMSensor() {
    while (PMSerial.available() > 0) {
        uint8_t byte = PMSerial.read();

        if (pmBufferIndex == 0 && byte == 0x42) {
            pmBuffer[pmBufferIndex++] = byte;
        }
        else if (pmBufferIndex == 1 && byte == 0x4D) {
            pmBuffer[pmBufferIndex++] = byte;
        }
        else if (pmBufferIndex >= 2 && pmBufferIndex < 32) {
            pmBuffer[pmBufferIndex++] = byte;

            if (pmBufferIndex == 32) {
                uint16_t checksum = 0;
                for (int i = 0; i < 30; i++) {
                    checksum += pmBuffer[i];
                }

                uint16_t receivedChecksum =
                    ((uint16_t)pmBuffer[30] << 8) | pmBuffer[31];

                if (checksum == receivedChecksum) {
                    pmData.pm1_0 = ((uint16_t)pmBuffer[10] << 8) | pmBuffer[11];
                    pmData.pm2_5 = ((uint16_t)pmBuffer[12] << 8) | pmBuffer[13];
                    pmData.pm10  = ((uint16_t)pmBuffer[14] << 8) | pmBuffer[15];
                    pmData.dataValid = true;
                    pmReady = true;

                    pmBufferIndex = 0;
                    return true;
                } else {
                    Serial.println("[PM] checksum fail");
                    pmBufferIndex = 0;
                }
            }
        }
        else {
            pmBufferIndex = 0;
        }
    }

    return false;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [8] 오도메트리 (★정밀 타이밍 및 가변 dt 동기화 패치)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void updateOdometry() {
    unsigned long now = millis();
    
    // 1. 계산 주기를 50ms (20Hz)로 칼같이 제한하여 라즈베리파이 송신 주기와 동기화합니다.
    if (now - lastOdomUpdate < 50) {
        return;
    }

    // 마이크로초 단위로 정밀한 dt 계산 (초 단위 변환)
    static unsigned long last_micros = 0;
    unsigned long current_micros = micros();
    if (last_micros == 0) {
        last_micros = current_micros;
        lastOdomUpdate = now;
        return;
    }
    
    float dt = (current_micros - last_micros) / 1000000.0f;
    last_micros = current_micros;

    // 예외 상황 방지 안전장치
    if (dt <= 0.0001f || dt > 0.5f) {
        lastOdomUpdate = now;
        return;
    }

    // 2. 바퀴 이동 거리 계산
    float distPerCount = (PI * WHEEL_DIAMETER) / (ENCODER_PPR * GEAR_RATIO);

    long d_enc1 = encoder1Count - lastEncoder1Count;
    long d_enc2 = encoder2Count - lastEncoder2Count;

    float leftDist  = d_enc1 * distPerCount;
    float rightDist = d_enc2 * distPerCount;

    float center = (leftDist + rightDist) / 2.0f;
    float deltaTheta = (rightDist - leftDist) / WHEEL_BASE;

    // 3. Runge-Kutta 적분을 이용한 바퀴축 중심 좌표계(w_odom) 누적 오차 최소화
    float midTheta = odomTheta + (deltaTheta / 2.0f);
    
    w_odomX += center * cos(midTheta);
    w_odomY += center * sin(midTheta);
    
    odomTheta += deltaTheta;
    
    // theta 각도 정규화 (-PI ~ PI)
    while (odomTheta > PI)  odomTheta -= 2.0f * PI;
    while (odomTheta < -PI) odomTheta += 2.0f * PI;

    // 4. 핵심 회전 오프셋 보정
    odomX = w_odomX - X_OFFSET * cos(odomTheta);
    odomY = w_odomY - X_OFFSET * sin(odomTheta);

    // 5. 정확한 물리적 선속도 및 각속도 계산
    linearVelocity  = center / dt;
    angularVelocity = deltaTheta / dt;

    // 히스토리 갱신
    lastEncoder1Count = encoder1Count;
    lastEncoder2Count = encoder2Count;
    lastOdomUpdate = now;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [9] 모터 제어
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void updateMotorRamping() {
    unsigned long now = millis();

    auto ramp = [&](MotorState &s, CytronMD &m, const char* name, bool isMotor1) {
        if (now - s.lastUpdate < RAMP_INTERVAL) {
            return;
        }

        if (s.currentSpeed < s.targetSpeed) {
            s.currentSpeed = min((int16_t)(s.currentSpeed + RAMP_STEP), s.targetSpeed);
        }
        else if (s.currentSpeed > s.targetSpeed) {
            s.currentSpeed = max((int16_t)(s.currentSpeed - RAMP_STEP), s.targetSpeed);
        }

        int16_t physicalSpeed;

        if (isMotor1) {
            physicalSpeed = toPhysicalMotor1Speed(s.currentSpeed);
        } else {
            physicalSpeed = toPhysicalMotor2Speed(s.currentSpeed);
        }

        m.setSpeed(physicalSpeed);

        s.lastUpdate = now;
    };

    ramp(motor1State, motor1, "M1", true);
    ramp(motor2State, motor2, "M2", false);
}

void setTargetSpeed(int16_t m1, int16_t m2) {
    motor1State.targetSpeed = constrain(m1, -255, 255);
    motor2State.targetSpeed = constrain(m2, -255, 255);
    lastMotorCommand = millis();
}

void emergencyStop() {
    motor1State.targetSpeed = 0;
    motor1State.currentSpeed = 0;

    motor2State.targetSpeed = 0;
    motor2State.currentSpeed = 0;

    motor1.setSpeed(0);
    motor2.setSpeed(0);

    Serial.println("Emergency Stop!");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [10] 팬 제어
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void updateFanControl() {
    unsigned long now = millis();


    // 20ms 주기로 팬 속도를 부드럽게 가감속 (램핑)
    if (now - lastFanUpdate >= 20) {
        if (fanSpeed < targetFanSpeed) {
            fanSpeed = min((uint8_t)(fanSpeed + 5), targetFanSpeed);
        }
        else if (fanSpeed > targetFanSpeed) {
            fanSpeed = max((uint8_t)(fanSpeed - 5), targetFanSpeed);
        }

        ledcWrite(FAN_PWM_CH, fanSpeed);
        // ★ 디버깅용: 1초마다 현재 팬 출력을 찍어보세요
        if (now % 1000 < 50) { 
             Serial.printf("DEBUG: target=%d, current_fanSpeed=%d\n", targetFanSpeed, fanSpeed);
        }
        lastFanUpdate = now;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [11] 공기질 센서 업데이트 (비동기 2초 주기로 제한하여 딜레이 방지)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
void updateAirSensors() {
    static unsigned long lastAirRead = 0;
    unsigned long now = millis();
    
    // DHT22의 블로킹 딜레이가 오도메트리를 방해하지 않도록 2초(2000ms)마다만 실행
    if (now - lastAirRead < 2000) {
        return;
    }
    lastAirRead = now;

    // DHT22 읽기
    float temp = dht.readTemperature();
    float humi = dht.readHumidity();

    if (!isnan(temp) && !isnan(humi)) {
        airTemperature = temp;
        airHumidity = humi;
        dhtReady = true;
    } else {
        dhtReady = false;
    }

    // SGP40
    if (sgp40Ready) {
        float compTemp = dhtReady ? airTemperature : 25.0f;
        float compHumi = dhtReady ? airHumidity : 50.0f;

        int32_t vocIndex = sgp40.measureVocIndex(compTemp, compHumi);
        vocValue = (vocIndex >= 0) ? (float)vocIndex : -1.0f;
    } else {
        vocValue = -1.0f;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [12] 모터 커맨드 수신
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void processCommands() {
    static uint8_t buf[32]; // 가장 긴 패킷도 담을 수 있게 버퍼 크기 증가
    static uint8_t idx = 0;

    while (Serial2.available()) {
        uint8_t b = Serial2.read();

        // 1. 헤더 검사
        if (idx == 0 && b == 0x55) {
            buf[idx++] = b;
        }
        else if (idx == 1 && b == 0xAA) {
            buf[idx++] = b;
        }
        else if (idx >= 2) {
            buf[idx++] = b;
            
            uint8_t current_id = buf[2]; // 수신된 패킷의 ID 확인

            // ----------------------------------------------------
            // 2. 바퀴 모터 명령 수신 (ID == 2) -> 질문자님 코드 그대로 적용
            // ----------------------------------------------------
            if (current_id == 2 && idx == sizeof(MotorCommandPacket)) {
                MotorCommandPacket* cmd = (MotorCommandPacket*)buf;
                uint8_t expectedChecksum = calculateChecksum(buf, sizeof(MotorCommandPacket));

                if (cmd->checksum == expectedChecksum) {
                    Serial.printf("[CMD RAW] mode=%d left=%d right=%d checksum=OK\n",
                                  cmd->mode, cmd->leftSpeed, cmd->rightSpeed);
                    currentMode = (DriveMode)cmd->mode;

                    if (currentMode == MODE_STOP) {
                        emergencyStop();
                    } else {
                        setTargetSpeed(cmd->leftSpeed, cmd->rightSpeed);
                    }
                } else {
                    Serial.println("[CMD] Motor checksum fail");
                }
                idx = 0; // 패킷 처리 완료 후 인덱스 초기화
            }
           // sizeof() 대신 명시적으로 7바이트를 기다리도록 변경
            else if (current_id == 4 && idx == 7) { 
                FanCommandPacket* cmd = (FanCommandPacket*)buf;
                
                // 계산할 때도 sizeof 대신 7을 사용
                uint8_t expectedChecksum = calculateChecksum(buf, 7);

                if (cmd->checksum == expectedChecksum) {
                    targetFanSpeed = cmd->fanSpeed;
                    Serial.printf("[CMD FAN] targetFanSpeed=%d checksum=OK\n", targetFanSpeed);
                } else {
                    Serial.println("[CMD] Fan checksum fail");
                    // 체크섬이 틀렸을 때 무엇이 들어왔는지 확인하면 디버깅이 빠릅니다.
                    Serial.printf("Received Chk: %d, Calculated: %d\n", cmd->checksum, expectedChecksum);
                }
                idx = 0; 
            }
            // ----------------------------------------------------
            // 4. 쓰레기 데이터가 들어와서 버퍼를 초과하면 초기화
            // ----------------------------------------------------
            else if (idx >= 32) {
                idx = 0;
            }
        }
        else {
            idx = 0;
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [13] IMU 초기화
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool initIMU() {
    Wire.beginTransmission(0x68);

    if (Wire.endTransmission() != 0) {
        Serial.println("✗ IMU: 응답 없음");
        return false;
    }

    if (imu.begin() < 0) {
        Serial.println("✗ IMU begin() 실패");
        return false;
    }

    Serial.println("✓ IMU 준비 (0x68)");
    return true;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [14] setup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void setup() {
    Serial.begin(115200);

    unsigned long t = millis();
    while (!Serial && millis() - t < 3000) {
        delay(10);
    }

    delay(500);

    // XSHUT 전부 LOW
    pinMode(TOF1_XSHUT, OUTPUT);
    pinMode(TOF2_XSHUT, OUTPUT);
    pinMode(TOF3_XSHUT, OUTPUT);

    digitalWrite(TOF1_XSHUT, LOW);
    digitalWrite(TOF2_XSHUT, LOW);
    digitalWrite(TOF3_XSHUT, LOW);

    // 모터 핀 플로팅 방지
    pinMode(MOTOR1_PWM, OUTPUT);
    pinMode(MOTOR1_DIR, OUTPUT);
    pinMode(MOTOR2_PWM, OUTPUT);
    pinMode(MOTOR2_DIR, OUTPUT);

    digitalWrite(MOTOR1_PWM, LOW);
    digitalWrite(MOTOR1_DIR, LOW);
    digitalWrite(MOTOR2_PWM, LOW);
    digitalWrite(MOTOR2_DIR, LOW);

    // Raspberry Pi 통신
    Serial2.begin(115200, SERIAL_8N1, RX2_PIN, TX2_PIN);

    // PM 센서 UART
    PMSerial.begin(9600, SERIAL_8N1, PM_RX_PIN, PM_TX_PIN);
    PMSerial.setTimeout(20);

    Serial.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.println("  ARIA 통합 펌웨어 시작");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // I2C
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(50000);
    Wire.setTimeOut(10);
    delay(100);

    i2cScan();

    // 엔코더
    pinMode(ENCODER1_A, INPUT_PULLUP);
    pinMode(ENCODER1_B, INPUT_PULLUP);
    pinMode(ENCODER2_A, INPUT_PULLUP);
    pinMode(ENCODER2_B, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(ENCODER1_A), encoder1_ISR, RISING);
    attachInterrupt(digitalPinToInterrupt(ENCODER2_A), encoder2_ISR, RISING);

    lastOdomUpdate = millis();
    // -------------------------------------------------------------
    // [수정] 초기화 시점에 내부 변수도 함께 세팅
    // -------------------------------------------------------------
    w_odomX = 0.0f;
    w_odomY = 0.0f;
    // -------------------------------------------------------------
    Serial.println("✓ 모터 & 엔코더 준비");

    // 팬
    pinMode(FAN_PIN, OUTPUT);
    ledcSetup(FAN_PWM_CH, FAN_PWM_FREQ, FAN_PWM_RES);
    ledcAttachPin(FAN_PIN, FAN_PWM_CH);
    ledcWrite(FAN_PWM_CH, 0);
    lastFanUpdate = millis();

    Serial.println("✓ 팬 PWM 준비");

    // IMU
    imuReady = initIMU();

    // ToF 초기화
    Serial.println("\n[ToF 초기화]");

    digitalWrite(TOF1_XSHUT, LOW);
    delay(200);
    digitalWrite(TOF2_XSHUT, LOW);
    delay(200);
    digitalWrite(TOF3_XSHUT, LOW);
    delay(200);

    // ToF1: VL53L0X, 0x30
    digitalWrite(TOF1_XSHUT, HIGH);
    delay(1000);

    if (!tof1.begin(0x29, false, &Wire)) {
        Serial.println("✗ ToF1 실패");
    } else {
        tof1.setAddress(0x30);
        tof1.startRangeContinuous(50);
        tof1Ready = true;
        Serial.println("✓ ToF1 준비 (VL53L0X 0x30)");
    }

    // ToF2: VL53L0X, 0x31
    digitalWrite(TOF2_XSHUT, HIGH);
    delay(1000);

    if (!tof2.begin(0x29, false, &Wire)) {
        Serial.println("✗ ToF2 실패");
    } else {
        tof2.setAddress(0x31);
        tof2.startRangeContinuous(50);
        tof2Ready = true;
        Serial.println("✓ ToF2 준비 (VL53L0X 0x31)");
    }

    // ★ [ToF3 사양 변경점] 기존 VL53L1X 로직을 걷어내고 ToF1, 2와 동일한 VL53L0X 로직으로 통일합니다.
    // 중복 주소를 피해 0x32 번지로 세팅합니다.
    digitalWrite(TOF3_XSHUT, HIGH);
    delay(1000);

    if (!tof3.begin(0x29, false, &Wire)) {
        Serial.println("✗ ToF3 실패");
    } else {
        tof3.setAddress(0x32);
        tof3.startRangeContinuous(50);
        tof3Ready = true;
        Serial.println("✓ ToF3 준비 (VL53L0X 0x32)");
    }

    // 공기질 센서 초기화
    Serial.println("\n[공기질 센서 초기화]");

    dht.begin();
    Serial.println("✓ DHT22 초기화 요청 완료");

    if (!sgp40.begin(&Wire)) {
        sgp40Ready = false;
        Serial.println("✗ SGP40 실패");
    } else {
        sgp40Ready = true;
        Serial.println("✓ SGP40 준비");
    }

    Serial.println("✓ PM 센서 UART 시작");
    Serial.printf("  PM_RX_PIN=%d, PM_TX_PIN=%d\n", PM_RX_PIN, PM_TX_PIN);

    i2cScan();

    Serial.println("\n[패킷 크기]");
    Serial.printf("  sizeof(NavPacket)=%d\n", (int)sizeof(NavPacket));
    Serial.printf("  sizeof(AirPacket)=%d\n", (int)sizeof(AirPacket));
    Serial.printf("  sizeof(OdomPacket)=%d\n", (int)sizeof(OdomPacket));
    Serial.printf("  sizeof(MotorCommandPacket)=%d\n", (int)sizeof(MotorCommandPacket));

    Serial.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.println("  시스템 준비 완료!");
    Serial.printf("  IMU:   %s\n", imuReady   ? "OK" : "FAIL");
    Serial.printf("  ToF1:  %s\n", tof1Ready  ? "OK" : "FAIL");
    Serial.printf("  ToF2:  %s\n", tof2Ready  ? "OK" : "FAIL");
    Serial.printf("  ToF3:  %s\n", tof3Ready  ? "OK" : "FAIL");
    Serial.printf("  SGP40: %s\n", sgp40Ready ? "OK" : "FAIL");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [15] loop (오도메트리 최우선 보장 + 타임 슬라이싱 센서 스케줄링 완벽 통합본)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
void loop() {
    unsigned long now = millis();

    // ──────────────────────────────────────────────────────────────
    // [최우선 순위] 바퀴 오도메트리 및 모터 제어 (매 루프마다 지연 없이 실행)
    // ──────────────────────────────────────────────────────────────
    updateOdometry();
    updateMotorRamping();
    processCommands();

    // 💡 [개선] 20ms(50Hz) 주기로 메인 송신 루프 외부에서 IMU 비동기 수집 (유지)
    if (now - lastImuRead >= 20) {
        if (imuReady && imu.readSensor() > 0) {
            latest_accelX = imu.getAccelX_mss();
            latest_accelY = imu.getAccelY_mss();
            latest_accelZ = imu.getAccelZ_mss();
            latest_gyroX  = imu.getGyroX_rads();
            latest_gyroY  = imu.getGyroY_rads();
            latest_gyroZ  = imu.getGyroZ_rads();
        }
        lastImuRead = now;
    }

    // [백그라운드] 수신 버퍼 수집 및 팬 제어 (비동기)
    readPMSensor();
    updateFanControl();

    // 모터 명령 타임아웃 검사
    if (now >= lastMotorCommand && (now - lastMotorCommand > MOTOR_TIMEOUT)) {
        if (motor1State.targetSpeed != 0 || motor2State.targetSpeed != 0) {
            motor1State.targetSpeed = 0;
            motor2State.targetSpeed = 0;
            Serial.println("[MOTOR TIMEOUT] target -> 0");
        }
    }

    // ──────────────────────────────────────────────────────────────
    // [송신 스케줄러] 각 패킷의 송신 주기를 독립적으로 제어하여 병목 방지
    // ──────────────────────────────────────────────────────────────

    // 1️⃣ OdomPacket 송신 — 30ms 주기 (약 33Hz) -> 최우선 보장
    if (now - lastOdom >= 30) {
        OdomPacket oPkt;
        oPkt.leftEncoderCount  = encoder1Count;
        oPkt.rightEncoderCount = encoder2Count;
        oPkt.posX       = odomX;
        oPkt.posY       = odomY;
        oPkt.posTheta   = odomTheta;
        oPkt.linearVel  = linearVelocity;
        oPkt.angularVel = angularVelocity;

        oPkt.checksum = calculateChecksum((uint8_t*)&oPkt, sizeof(oPkt));
        Serial2.write((uint8_t*)&oPkt, sizeof(oPkt));
        lastOdom = now;
    }

    // 2️⃣ NavPacket (ToF 데이터 융합) 송신 — 100ms 주기 (10Hz)
    // 💡 [핵심 수정] readRangeResult가 완전히 완료되었을 때만 가져오도록 non-blocking 구조 강화
    if (now - lastNav >= 100) {
        NavPacket nPkt;

        // 대각선/정면 ToF 센서 상호 간섭 방지 및 블로킹 제거를 위해 isRangeComplete 확인 후 수집
        // 연속 모드이므로 무작정 delay를 주기보다 값이 준비되었는지 체크하여 I2C 멈춤 현상 차단
        nPkt.tof1Distance = (tof1Ready && tof1.isRangeComplete()) ? readTof0xDistance(tof1) : -1.0f;
        nPkt.tof2Distance = (tof2Ready && tof2.isRangeComplete()) ? readTof0xDistance(tof2) : -1.0f;
        nPkt.tof3Distance = (tof3Ready && tof3.isRangeComplete()) ? readTof0xDistance(tof3) : -1.0f;
        
        // 실시간으로 갱신해 둔 전역 변수 값 대입 (블로킹 0초)
        nPkt.accelX = latest_accelX;
        nPkt.accelY = latest_accelY;
        nPkt.accelZ = latest_accelZ;
        nPkt.gyroX  = latest_gyroX;
        nPkt.gyroY  = latest_gyroY;
        nPkt.gyroZ  = latest_gyroZ;
        
        nPkt.checksum = calculateChecksum((uint8_t*)&nPkt, sizeof(nPkt));
        Serial2.write((uint8_t*)&nPkt, sizeof(nPkt));
        lastNav = now;
    }

    // 3️⃣ AirPacket 송신 및 센서 읽기 — 2000ms 주기 (0.5Hz)
    // 💡 [핵심 수정] DHT22의 악명 높은 블로킹 delay(몇십ms)가 odom 주기를 터뜨리지 못하도록 
    // updateAirSensors() 호출 자체를 이 2초 주기 안으로 격리시켰습니다.
    if (now - lastAir >= 2000) {
        // 2초에 한 번만 온습도/공기질 센서를 물리적으로 읽음 (메인 루프와 완벽 분리)
        updateAirSensors(); 

        AirPacket aPkt;
        aPkt.pm25        = pmReady ? (float)pmData.pm2_5 : -1.0f;
        aPkt.voc         = vocValue;
        aPkt.temperature = airTemperature;
        aPkt.humidity    = airHumidity;

        aPkt.checksum = calculateChecksum((uint8_t*)&aPkt, sizeof(aPkt));
        Serial2.write((uint8_t*)&aPkt, sizeof(aPkt));
        lastAir = now;
    }
}

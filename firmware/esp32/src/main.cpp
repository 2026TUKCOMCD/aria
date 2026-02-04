/*
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * 2륜 자율주행 공기청정기 로봇 - ESP32 메인 펌웨어
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 */

#include <Arduino.h>
#include <Wire.h>
#include "Adafruit_VL53L1X.h"
#include "mpu9250.h"         
#include "Adafruit_SGP40.h"
#include "CytronMotorDriver.h"
#include "DHT.h"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [1] 핀 설정
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#define TX2_PIN 17
#define RX2_PIN 16
#define I2C_SDA 21
#define I2C_SCL 22
#define TOF1_XSHUT 26
#define TOF2_XSHUT 27

#define MOTOR1_PWM 25
#define MOTOR1_DIR 33
#define MOTOR2_PWM 32
#define MOTOR2_DIR 14

#define ENCODER1_A 18
#define ENCODER1_B 19
#define ENCODER2_A 23
#define ENCODER2_B 5

#define DHT_PIN 4
#define DHT_TYPE DHT22

#define PM_RX_PIN 15       // PM 센서 TX → ESP32 RX
#define PM_TX_PIN 2        // ESP32 TX → PM 센서 RX

// 팬 제어 (MOSFET)
#define FAN_PIN 13         // MOSFET Gate 핀
#define FAN_PWM_CHANNEL 2  // PWM 채널 (0, 1은 모터 사용)
#define FAN_PWM_FREQ 25000 // 25kHz (팬에 최적)
#define FAN_PWM_RES 8      // 8bit (0~255)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [2] 로봇 파라미터
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#define ENCODER_PPR 360
#define WHEEL_DIAMETER 0.065
#define WHEEL_BASE 0.15

#define RAMP_STEP 5
#define RAMP_INTERVAL 20
#define MOTOR_TIMEOUT 1000

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [3] 통신 패킷 구조체
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma pack(push, 1)

struct NavPacket {
    uint16_t header = 0xAA55;
    uint8_t id = 0;
    float tof1Distance;
    float tof2Distance;
    float accelX;
    float accelY;
    float accelZ;
    float gyroX;
    float gyroY;
    float gyroZ;
    uint8_t checksum;
    uint16_t tail = 0x0A0D;
};

struct AirPacket {
    uint16_t header = 0xAA55;
    uint8_t id = 1;
    float pm25;
    float vocIndex;
    float temperature;
    float humidity;
    uint8_t checksum;
    uint16_t tail = 0x0A0D;
};

struct OdomPacket {
    uint16_t header = 0xAA55;
    uint8_t id = 3;
    int32_t leftEncoderCount;
    int32_t rightEncoderCount;
    float posX;
    float posY;
    float posTheta;
    float linearVel;
    float angularVel;
    uint8_t checksum;
    uint16_t tail = 0x0A0D;
};

struct MotorCommandPacket {
    uint16_t header = 0xAA55;
    uint8_t id = 2;
    uint8_t mode;
    int16_t leftSpeed;
    int16_t rightSpeed;
    uint8_t checksum;
    uint16_t tail = 0x0A0D;
};

#pragma pack(pop)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [4] 전역 객체 및 변수
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Adafruit_VL53L1X tof1, tof2; 
MPU9250 imu(Wire, 0x68);
Adafruit_SGP40 sgp;
DHT dht(DHT_PIN, DHT_TYPE);

CytronMD motor1(PWM_DIR, MOTOR1_PWM, MOTOR1_DIR);
CytronMD motor2(PWM_DIR, MOTOR2_PWM, MOTOR2_DIR);

struct MotorState {
    int16_t targetSpeed;
    int16_t currentSpeed;
    unsigned long lastUpdate;
};

enum DriveMode {
    MODE_STOP = 0,
    MODE_MANUAL = 1,
    MODE_AUTO = 2
};

MotorState motor1State = {0, 0, 0};
MotorState motor2State = {0, 0, 0};
DriveMode currentMode = MODE_STOP;

volatile int32_t encoder1Count = 0;
volatile int32_t encoder2Count = 0;
int32_t lastEncoder1Count = 0;
int32_t lastEncoder2Count = 0;

float odomX = 0.0;
float odomY = 0.0;
float odomTheta = 0.0;
float linearVelocity = 0.0;
float angularVelocity = 0.0;
unsigned long lastOdomUpdate = 0;

unsigned long lastNav = 0;
unsigned long lastAir = 0;
unsigned long lastOdom = 0;
unsigned long lastMotorCommand = 0;

// PM 센서 데이터 구조체
struct PMSensorData {
    uint16_t pm1_0;
    uint16_t pm2_5;
    uint16_t pm10;
    bool dataValid;
};

PMSensorData pmData = {0, 0, 0, false};
uint8_t pmBuffer[32];
uint8_t pmBufferIndex = 0;

// 팬 제어 변수
uint8_t fanSpeed = 0;           // 현재 팬 속도 (0~255)
uint8_t targetFanSpeed = 0;     // 목표 팬 속도
unsigned long lastFanUpdate = 0;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [5] 엔코더 인터럽트
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void IRAM_ATTR encoder1_ISR() {
    int b = digitalRead(ENCODER1_B);
    if (b > 0) encoder1Count++;
    else encoder1Count--;
}

void IRAM_ATTR encoder2_ISR() {
    int b = digitalRead(ENCODER2_B);
    if (b > 0) encoder2Count++;
    else encoder2Count--;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [6] PM 센서 UART 처리
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool readPMSensor() {
    while (Serial1.available() > 0) {
        uint8_t byte = Serial1.read();
        
        if (pmBufferIndex == 0 && byte == 0x42) {
            pmBuffer[pmBufferIndex++] = byte;
        }
        else if (pmBufferIndex == 1 && byte == 0x4D) {
            pmBuffer[pmBufferIndex++] = byte;
        }
        else if (pmBufferIndex >= 2 && pmBufferIndex < 32) {
            pmBuffer[pmBufferIndex++] = byte;
            
            if (pmBufferIndex == 32) {
                // 체크섬 검증
                uint16_t checksum = 0;
                for (int i = 0; i < 30; i++) {
                    checksum += pmBuffer[i];
                }
                uint16_t receivedChecksum = (pmBuffer[30] << 8) | pmBuffer[31];
                
                if (checksum == receivedChecksum) {
                    pmData.pm1_0 = (pmBuffer[10] << 8) | pmBuffer[11];
                    pmData.pm2_5 = (pmBuffer[12] << 8) | pmBuffer[13];
                    pmData.pm10 = (pmBuffer[14] << 8) | pmBuffer[15];
                    pmData.dataValid = true;
                    
                    pmBufferIndex = 0;
                    return true;
                } else {
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
// [7] 오도메트리 계산
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void updateOdometry() {
    unsigned long now = millis();
    float dt = (now - lastOdomUpdate) / 1000.0;
    
    if (dt < 0.001) return;
    
    int32_t deltaEncoder1 = encoder1Count - lastEncoder1Count;
    int32_t deltaEncoder2 = encoder2Count - lastEncoder2Count;
    
    float distancePerCount = (PI * WHEEL_DIAMETER) / ENCODER_PPR;
    float leftDistance = deltaEncoder1 * distancePerCount;
    float rightDistance = deltaEncoder2 * distancePerCount;
    
    float centerDistance = (leftDistance + rightDistance) / 2.0;
    float deltaTheta = (rightDistance - leftDistance) / WHEEL_BASE;
    
    odomX += centerDistance * cos(odomTheta + deltaTheta / 2.0);
    odomY += centerDistance * sin(odomTheta + deltaTheta / 2.0);
    odomTheta += deltaTheta;
    
    while (odomTheta > PI) odomTheta -= 2 * PI;
    while (odomTheta < -PI) odomTheta += 2 * PI;
    
    linearVelocity = centerDistance / dt;
    angularVelocity = deltaTheta / dt;
    
    lastEncoder1Count = encoder1Count;
    lastEncoder2Count = encoder2Count;
    lastOdomUpdate = now;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [8] 유틸리티
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

uint8_t calculateChecksum(uint8_t* data, size_t len) {
    uint8_t sum = 0;
    for (size_t i = 0; i < len - 3; i++) {
        sum += data[i];
    }
    return sum;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [9] 모터 제어
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void updateMotorRamping() {
    unsigned long now = millis();
    
    if (now - motor1State.lastUpdate >= RAMP_INTERVAL) {
        if (motor1State.currentSpeed < motor1State.targetSpeed) {
            motor1State.currentSpeed += RAMP_STEP;
            if (motor1State.currentSpeed > motor1State.targetSpeed) {
                motor1State.currentSpeed = motor1State.targetSpeed;
            }
        } else if (motor1State.currentSpeed > motor1State.targetSpeed) {
            motor1State.currentSpeed -= RAMP_STEP;
            if (motor1State.currentSpeed < motor1State.targetSpeed) {
                motor1State.currentSpeed = motor1State.targetSpeed;
            }
        }
        motor1.setSpeed(motor1State.currentSpeed);
        motor1State.lastUpdate = now;
    }
    
    if (now - motor2State.lastUpdate >= RAMP_INTERVAL) {
        if (motor2State.currentSpeed < motor2State.targetSpeed) {
            motor2State.currentSpeed += RAMP_STEP;
            if (motor2State.currentSpeed > motor2State.targetSpeed) {
                motor2State.currentSpeed = motor2State.targetSpeed;
            }
        } else if (motor2State.currentSpeed > motor2State.targetSpeed) {
            motor2State.currentSpeed -= RAMP_STEP;
            if (motor2State.currentSpeed < motor2State.targetSpeed) {
                motor2State.currentSpeed = motor2State.targetSpeed;
            }
        }
        motor2.setSpeed(motor2State.currentSpeed);
        motor2State.lastUpdate = now;
    }
}

void setTargetSpeed(int16_t m1Speed, int16_t m2Speed) {
    if (m1Speed > 255) m1Speed = 255;
    if (m1Speed < -255) m1Speed = -255;
    if (m2Speed > 255) m2Speed = 255;
    if (m2Speed < -255) m2Speed = -255;
    
    motor1State.targetSpeed = m1Speed;
    motor2State.targetSpeed = m2Speed;
    lastMotorCommand = millis();
}

void emergencyStop() {
    motor1State.targetSpeed = 0;
    motor2State.targetSpeed = 0;
    motor1State.currentSpeed = 0;
    motor2State.currentSpeed = 0;
    motor1.setSpeed(0);
    motor2.setSpeed(0);
    Serial.println("Emergency Stop!");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [9-2] 팬 제어
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void setFanSpeed(uint8_t speed) {
    /**
     * 팬 속도 설정
     * 
     * @param speed: 0~255 (0=정지, 255=최대)
     */
    targetFanSpeed = speed;
}

void updateFanControl() {
    /**
     * 팬 제어 로직 (자동 조절)
     * 
     * PM2.5 농도에 따라 팬 속도 자동 조절:
     * - 0~15:   30% (최소 청정)
     * - 16~35:  50% (보통 청정)
     * - 36~75:  80% (강력 청정)
     * - 76+:   100% (최대 청정)
     */
    unsigned long now = millis();
    
    // 2초마다 팬 속도 조절
    if (now - lastFanUpdate >= 2000) {
        if (pmData.dataValid) {
            float pm25 = pmData.pm2_5;
            
            if (pm25 < 15) {
                targetFanSpeed = 77;   // 30% (77/255)
            } else if (pm25 < 35) {
                targetFanSpeed = 128;  // 50%
            } else if (pm25 < 75) {
                targetFanSpeed = 204;  // 80%
            } else {
                targetFanSpeed = 255;  // 100%
            }
        } else {
            // PM 센서 데이터 없으면 기본 속도
            targetFanSpeed = 128;  // 50%
        }
        
        lastFanUpdate = now;
    }
    
    // 부드러운 팬 속도 변경 (급변 방지)
    if (fanSpeed < targetFanSpeed) {
        fanSpeed += 5;
        if (fanSpeed > targetFanSpeed) fanSpeed = targetFanSpeed;
    } else if (fanSpeed > targetFanSpeed) {
        fanSpeed -= 5;
        if (fanSpeed < targetFanSpeed) fanSpeed = targetFanSpeed;
    }
    
    // PWM 출력
    ledcWrite(FAN_PWM_CHANNEL, fanSpeed);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [10] 통신
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void processMotorCommand() {
    static uint8_t buffer[sizeof(MotorCommandPacket)];
    static uint8_t bufferIndex = 0;
    
    while (Serial2.available() > 0) {
        uint8_t byte = Serial2.read();
        
        if (bufferIndex == 0 && byte == 0x55) {
            buffer[bufferIndex++] = byte;
        } else if (bufferIndex == 1 && byte == 0xAA) {
            buffer[bufferIndex++] = byte;
        } else if (bufferIndex >= 2 && bufferIndex < sizeof(MotorCommandPacket)) {
            buffer[bufferIndex++] = byte;
            
            if (bufferIndex == sizeof(MotorCommandPacket)) {
                MotorCommandPacket* cmd = (MotorCommandPacket*)buffer;
                
                if (cmd->id == 2) {
                    uint8_t expectedChecksum = calculateChecksum(buffer, sizeof(MotorCommandPacket));
                    if (cmd->checksum == expectedChecksum) {
                        currentMode = (DriveMode)cmd->mode;
                        
                        if (currentMode == MODE_STOP) {
                            emergencyStop();
                        } else {
                            setTargetSpeed(cmd->leftSpeed, cmd->rightSpeed);
                        }
                    }
                }
                bufferIndex = 0;
            }
        } else {
            bufferIndex = 0;
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [11] 초기화
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void setup() {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ⭐ 최우선: 모터 핀 초기화 (플로팅 방지)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pinMode(MOTOR1_PWM, OUTPUT);
    pinMode(MOTOR1_DIR, OUTPUT);
    pinMode(MOTOR2_PWM, OUTPUT);
    pinMode(MOTOR2_DIR, OUTPUT);
    
    digitalWrite(MOTOR1_PWM, LOW);
    digitalWrite(MOTOR1_DIR, LOW);
    digitalWrite(MOTOR2_PWM, LOW);
    digitalWrite(MOTOR2_DIR, LOW);
    
    delay(100);  // 안정화 대기
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 통신 초기화
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Serial.begin(115200);
    Serial2.begin(115200, SERIAL_8N1, RX2_PIN, TX2_PIN);
    Serial1.begin(9600, SERIAL_8N1, PM_RX_PIN, PM_TX_PIN);  // PM 센서
    
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);
    
    Serial.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.println("  ESP32 펌웨어 시작");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 모터 및 엔코더
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pinMode(ENCODER1_A, INPUT_PULLUP);
    pinMode(ENCODER1_B, INPUT_PULLUP);
    pinMode(ENCODER2_A, INPUT_PULLUP);
    pinMode(ENCODER2_B, INPUT_PULLUP);
    
    attachInterrupt(digitalPinToInterrupt(ENCODER1_A), encoder1_ISR, RISING);
    attachInterrupt(digitalPinToInterrupt(ENCODER2_A), encoder2_ISR, RISING);
    
    Serial.println("✓ 모터 & 엔코더 준비 (핀 초기화 완료)");
    lastOdomUpdate = millis();
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 팬 제어 초기화
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pinMode(FAN_PIN, OUTPUT);
    ledcSetup(FAN_PWM_CHANNEL, FAN_PWM_FREQ, FAN_PWM_RES);
    ledcAttachPin(FAN_PIN, FAN_PWM_CHANNEL);
    ledcWrite(FAN_PWM_CHANNEL, 0);  // 초기: 정지
    Serial.println("✓ 팬 제어 준비 (MOSFET)");
    lastFanUpdate = millis();
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // IMU
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (imu.begin() < 0) {
        Serial.println("✗ IMU 연결 실패");
    } else {
        Serial.println("✓ IMU 준비 (가속도+자이로)");
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ToF 센서
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pinMode(TOF1_XSHUT, OUTPUT);
    pinMode(TOF2_XSHUT, OUTPUT);
    digitalWrite(TOF1_XSHUT, LOW);
    digitalWrite(TOF2_XSHUT, LOW);
    delay(10);
    
    digitalWrite(TOF1_XSHUT, HIGH);
    delay(10);
    if (!tof1.begin(0x29, &Wire)) {
        Serial.println("✗ ToF1 연결 실패");
    } else {
        tof1.VL53L1X_SetI2CAddress(0x30);
        tof1.startRanging();
        Serial.println("✓ ToF1 준비 (0x30)");
    }
    
    digitalWrite(TOF2_XSHUT, HIGH);
    delay(10);
    if (!tof2.begin(0x29, &Wire)) {
        Serial.println("✗ ToF2 연결 실패");
    } else {
        tof2.startRanging();
        Serial.println("✓ ToF2 준비 (0x29)");
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 환경 센서
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Serial.println("✓ PM 센서 준비 (UART)");
    Serial.println("  ⚠ SET 핀을 GND에 연결!");
    
    if (!sgp.begin()) {
        Serial.println("✗ SGP40 연결 실패");
    } else {
        Serial.println("✓ SGP40 준비 (VOC)");
    }
    
    dht.begin();
    Serial.println("✓ DHT22 준비 (온습도)");
    
    Serial.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.println("  시스템 준비 완료!");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [12] 메인 루프
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void loop() {
    unsigned long now = millis();
    
    updateOdometry();
    updateMotorRamping();
    processMotorCommand();
    readPMSensor();  // 백그라운드 수신
    updateFanControl();  // 팬 자동 제어 ⭐
    
    // 타임아웃 체크
    if (now - lastMotorCommand > MOTOR_TIMEOUT) {
        if (motor1State.targetSpeed != 0 || motor2State.targetSpeed != 0) {
            setTargetSpeed(0, 0);
        }
    }
    
    // OdomPacket - 50ms
    if (now - lastOdom >= 50) {
        OdomPacket oPkt;
        oPkt.leftEncoderCount = encoder1Count;
        oPkt.rightEncoderCount = encoder2Count;
        oPkt.posX = odomX;
        oPkt.posY = odomY;
        oPkt.posTheta = odomTheta;
        oPkt.linearVel = linearVelocity;
        oPkt.angularVel = angularVelocity;
        oPkt.checksum = calculateChecksum((uint8_t*)&oPkt, sizeof(oPkt));
        Serial2.write((uint8_t*)&oPkt, sizeof(oPkt));
        lastOdom = now;
    }
    
    // NavPacket - 50ms
    if (now - lastNav >= 50) {
        NavPacket nPkt;
        
        if (tof1.dataReady()) { 
            nPkt.tof1Distance = (float)tof1.distance(); 
            tof1.clearInterrupt(); 
        }
        if (tof2.dataReady()) { 
            nPkt.tof2Distance = (float)tof2.distance(); 
            tof2.clearInterrupt(); 
        }
        
        if (imu.readSensor() > 0) {
            nPkt.accelX = imu.getAccelX_mss();
            nPkt.accelY = imu.getAccelY_mss();
            nPkt.accelZ = imu.getAccelZ_mss();
            nPkt.gyroX = imu.getGyroX_rads();
            nPkt.gyroY = imu.getGyroY_rads();
            nPkt.gyroZ = imu.getGyroZ_rads();
        }
        
        nPkt.checksum = calculateChecksum((uint8_t*)&nPkt, sizeof(nPkt));
        Serial2.write((uint8_t*)&nPkt, sizeof(nPkt));
        lastNav = now;
    }
    
    // AirPacket - 2000ms
    if (now - lastAir >= 2000) {
        AirPacket aPkt;
        
        // PM 센서 (UART)
        if (pmData.dataValid) {
            aPkt.pm25 = (float)pmData.pm2_5;
            pmData.dataValid = false;
        } else {
            aPkt.pm25 = -1.0f;
        }
        
        // VOC
        int32_t voc_index = sgp.measureVocIndex();
        aPkt.vocIndex = (voc_index >= 0) ? (float)voc_index : -1.0f;
        
        // 온습도
        float temp = dht.readTemperature();
        float humi = dht.readHumidity();
        aPkt.temperature = (!isnan(temp)) ? temp : -999.0f;
        aPkt.humidity = (!isnan(humi)) ? humi : -1.0f;
        
        aPkt.checksum = calculateChecksum((uint8_t*)&aPkt, sizeof(aPkt));
        Serial2.write((uint8_t*)&aPkt, sizeof(aPkt));
        lastAir = now;
    }
}
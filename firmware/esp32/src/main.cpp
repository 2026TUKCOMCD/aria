/*
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * 2륜 자율주행 공기청정기 로봇 - ESP32 메인 펌웨어
 * [모터드라이버 + 엔코더 + IMU + ToF 1개 + 라즈베리파이 통신]
 * [SGP40 비활성화]
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 */

#include <Arduino.h>
#include <Wire.h>
<<<<<<< HEAD
#include "Adafruit_VL53L1X.h"
#include "mpu9250.h"
=======
#include "Adafruit_VL53L0X.h"
#include "mpu9250.h"         
#include "Adafruit_SGP40.h"
>>>>>>> 11d1f6a4052fbd19a58cecfd155ef7a384317549
#include "CytronMotorDriver.h"
// #include "Adafruit_SGP40.h"
// #include "DHT.h"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [1] 핀 설정
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#define TX2_PIN 17
#define RX2_PIN 16
#define I2C_SDA 21
#define I2C_SCL 22

#define TOF1_XSHUT 26
<<<<<<< HEAD
#define TOF2_XSHUT 27   // 고장난 ToF는 비활성화
=======
#define TOF2_XSHUT 27
#define TOF3_XSHUT 12  // GPIO12: 부팅 시 LOW 유지 필요 → 10K 풀다운 저항 필수
>>>>>>> 11d1f6a4052fbd19a58cecfd155ef7a384317549

#define MOTOR1_PWM 25
#define MOTOR1_DIR 33
#define MOTOR2_PWM 32
#define MOTOR2_DIR 14

#define ENCODER1_A 18
#define ENCODER1_B 19
#define ENCODER2_A 23
#define ENCODER2_B 5

// #define DHT_PIN 4
// #define DHT_TYPE DHT22

// #define PM_RX_PIN 15
// #define PM_TX_PIN 2

// #define FAN_PIN 13
// #define FAN_PWM_CHANNEL 2
// #define FAN_PWM_FREQ 25000
// #define FAN_PWM_RES 8

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [2] 로봇 파라미터
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#define ENCODER_PPR 16
#define WHEEL_DIAMETER 0.095
#define WHEEL_BASE 0.23

#define RAMP_STEP 5
#define RAMP_INTERVAL 20
#define MOTOR_TIMEOUT 10000

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [3] 통신 패킷 구조체
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma pack(push, 1)

struct NavPacket {
    uint16_t header = 0xAA55;
    uint8_t id = 0;
    float tof1Distance;
    float tof2Distance;
    float tof3Distance;
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

<<<<<<< HEAD
Adafruit_VL53L1X tof1 = Adafruit_VL53L1X(TOF1_XSHUT, -1);
=======
Adafruit_VL53L0X tof1, tof2, tof3;
>>>>>>> 11d1f6a4052fbd19a58cecfd155ef7a384317549
MPU9250 imu(Wire, 0x68);
// Adafruit_SGP40 sgp;
// DHT dht(DHT_PIN, DHT_TYPE);

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

float tof1DistanceValue = -1.0f;

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
// [6] 오도메트리 계산
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
// [7] 유틸리티
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

uint8_t calculateChecksum(uint8_t* data, size_t len) {
    uint8_t sum = 0;
    for (size_t i = 0; i < len - 3; i++) {
        sum += data[i];
    }
    return sum;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [8] 모터 제어
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void updateMotorRamping() {
    unsigned long now = millis();

    if (now - motor1State.lastUpdate >= RAMP_INTERVAL) {
        if (motor1State.currentSpeed < motor1State.targetSpeed) {
            motor1State.currentSpeed += RAMP_STEP;
            if (motor1State.currentSpeed > motor1State.targetSpeed)
                motor1State.currentSpeed = motor1State.targetSpeed;
        } else if (motor1State.currentSpeed > motor1State.targetSpeed) {
            motor1State.currentSpeed -= RAMP_STEP;
            if (motor1State.currentSpeed < motor1State.targetSpeed)
                motor1State.currentSpeed = motor1State.targetSpeed;
        }
        motor1.setSpeed(motor1State.currentSpeed);
        motor1State.lastUpdate = now;
    }

    if (now - motor2State.lastUpdate >= RAMP_INTERVAL) {
        if (motor2State.currentSpeed < motor2State.targetSpeed) {
            motor2State.currentSpeed += RAMP_STEP;
            if (motor2State.currentSpeed > motor2State.targetSpeed)
                motor2State.currentSpeed = motor2State.targetSpeed;
        } else if (motor2State.currentSpeed > motor2State.targetSpeed) {
            motor2State.currentSpeed -= RAMP_STEP;
            if (motor2State.currentSpeed < motor2State.targetSpeed)
                motor2State.currentSpeed = motor2State.targetSpeed;
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
// [9] 통신
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
// [10] 초기화
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void setup() {
    pinMode(MOTOR1_PWM, OUTPUT);
    pinMode(MOTOR1_DIR, OUTPUT);
    pinMode(MOTOR2_PWM, OUTPUT);
    pinMode(MOTOR2_DIR, OUTPUT);

    digitalWrite(MOTOR1_PWM, LOW);
    digitalWrite(MOTOR1_DIR, LOW);
    digitalWrite(MOTOR2_PWM, LOW);
    digitalWrite(MOTOR2_DIR, LOW);

    delay(100);

    Serial.begin(115200);
    Serial2.begin(115200, SERIAL_8N1, RX2_PIN, TX2_PIN);

    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);

    Serial.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.println("  ESP32 펌웨어 시작 (모터+엔코더+IMU+ToF1 모드)");
    Serial.println("  SGP40 비활성화");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    pinMode(ENCODER1_A, INPUT_PULLUP);
    pinMode(ENCODER1_B, INPUT_PULLUP);
    pinMode(ENCODER2_A, INPUT_PULLUP);
    pinMode(ENCODER2_B, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(ENCODER1_A), encoder1_ISR, RISING);
    attachInterrupt(digitalPinToInterrupt(ENCODER2_A), encoder2_ISR, RISING);

    Serial.println("✓ 모터 & 엔코더 준비");
    lastOdomUpdate = millis();

    // TOF1만 사용
    pinMode(TOF1_XSHUT, OUTPUT);
    pinMode(TOF2_XSHUT, OUTPUT);

    digitalWrite(TOF1_XSHUT, LOW);
    digitalWrite(TOF2_XSHUT, LOW);
    delay(50);

    digitalWrite(TOF1_XSHUT, HIGH);
    digitalWrite(TOF2_XSHUT, LOW);
    delay(50);

    if (!tof1.begin(0x29, &Wire)) {
        Serial.println("✗ TOF1 연결 실패");
    } else {
        Serial.println("✓ TOF1 준비");
        tof1.startRanging();
    }

    // IMU 초기화
    if (imu.begin() < 0) {
        Serial.println("✗ IMU 연결 실패");
    } else {
        Serial.println("✓ IMU 준비 (가속도+자이로)");
    }

    Serial.printf("sizeof(NavPacket)=%d\n", (int)sizeof(NavPacket));
    Serial.printf("sizeof(AirPacket)=%d\n", (int)sizeof(AirPacket));
    Serial.printf("sizeof(OdomPacket)=%d\n", (int)sizeof(OdomPacket));
    Serial.printf("sizeof(MotorCommandPacket)=%d\n", (int)sizeof(MotorCommandPacket));
<<<<<<< HEAD

=======
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ToF 센서 (VL53L0X x3) — 하나씩 깨워서 I2C 주소 분리
    //   tof1 → 0x30 / tof2 → 0x31 / tof3 → 0x29(기본)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pinMode(TOF1_XSHUT, OUTPUT);
    pinMode(TOF2_XSHUT, OUTPUT);
    pinMode(TOF3_XSHUT, OUTPUT);
    digitalWrite(TOF1_XSHUT, LOW);
    digitalWrite(TOF2_XSHUT, LOW);
    digitalWrite(TOF3_XSHUT, LOW);
    delay(10);

    // tof1: 0x29 → 0x30
    digitalWrite(TOF1_XSHUT, HIGH);
    delay(10);
    if (!tof1.begin(0x29, false, &Wire)) {
        Serial.println("✗ ToF1 연결 실패");
    } else {
        tof1.setAddress(0x30);
        Serial.println("✓ ToF1 준비 (0x30)");
    }

    // tof2: 0x29 → 0x31
    digitalWrite(TOF2_XSHUT, HIGH);
    delay(10);
    if (!tof2.begin(0x29, false, &Wire)) {
        Serial.println("✗ ToF2 연결 실패");
    } else {
        tof2.setAddress(0x31);
        Serial.println("✓ ToF2 준비 (0x31)");
    }

    // tof3: 0x29 유지
    digitalWrite(TOF3_XSHUT, HIGH);
    delay(10);
    if (!tof3.begin(0x29, false, &Wire)) {
        Serial.println("✗ ToF3 연결 실패");
    } else {
        Serial.println("✓ ToF3 준비 (0x29)");
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
    
>>>>>>> 11d1f6a4052fbd19a58cecfd155ef7a384317549
    Serial.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.println("  시스템 준비 완료!");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [11] 메인 루프
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void loop() {
    unsigned long now = millis();

    updateOdometry();
    updateMotorRamping();
    processMotorCommand();

    if (now - lastMotorCommand > MOTOR_TIMEOUT) {
        if (motor1State.targetSpeed != 0 || motor2State.targetSpeed != 0) {
            setTargetSpeed(0, 0);
        }
    }

    // TOF1 읽기
    if (tof1.dataReady()) {
        int distance = tof1.distance();
        if (distance > 0) {
            tof1DistanceValue = (float)distance / 1000.0f;  // mm -> m
        } else {
            tof1DistanceValue = -1.0f;
        }
        tof1.clearInterrupt();
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
<<<<<<< HEAD

        nPkt.tof1Distance = tof1DistanceValue;
        nPkt.tof2Distance = -1.0f;

=======
        
        VL53L0X_RangingMeasurementData_t measure;

        tof1.rangingTest(&measure, false);
        if (measure.RangeStatus != 4) nPkt.tof1Distance = (float)measure.RangeMilliMeter;

        tof2.rangingTest(&measure, false);
        if (measure.RangeStatus != 4) nPkt.tof2Distance = (float)measure.RangeMilliMeter;

        tof3.rangingTest(&measure, false);
        if (measure.RangeStatus != 4) nPkt.tof3Distance = (float)measure.RangeMilliMeter;
        
>>>>>>> 11d1f6a4052fbd19a58cecfd155ef7a384317549
        if (imu.readSensor() > 0) {
            nPkt.accelX = imu.getAccelX_mss();
            nPkt.accelY = imu.getAccelY_mss();
            nPkt.accelZ = imu.getAccelZ_mss();
            nPkt.gyroX  = imu.getGyroX_rads();
            nPkt.gyroY  = imu.getGyroY_rads();
            nPkt.gyroZ  = imu.getGyroZ_rads();
        } else {
            nPkt.accelX = 0.0f;
            nPkt.accelY = 0.0f;
            nPkt.accelZ = 0.0f;
            nPkt.gyroX  = 0.0f;
            nPkt.gyroY  = 0.0f;
            nPkt.gyroZ  = 0.0f;
        }

        nPkt.checksum = calculateChecksum((uint8_t*)&nPkt, sizeof(nPkt));
        Serial2.write((uint8_t*)&nPkt, sizeof(nPkt));
        lastNav = now;
    }

    // AirPacket - 2000ms
    // SGP40 비활성화 → 더미값 전송
    if (now - lastAir >= 2000) {
        AirPacket aPkt;

        aPkt.pm25        = -1.0f;
        aPkt.vocIndex    = -1.0f;
        aPkt.temperature = -999.0f;
        aPkt.humidity    = -1.0f;

        aPkt.checksum = calculateChecksum((uint8_t*)&aPkt, sizeof(aPkt));
        Serial2.write((uint8_t*)&aPkt, sizeof(aPkt));
        lastAir = now;
    }
}
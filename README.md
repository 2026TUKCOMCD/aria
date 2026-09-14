# ARIA: AI 기반 자율주행 로봇 공기청정기

![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=flat-square&logo=ros)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python)
![C++](https://img.shields.io/badge/C++-17-00599C?style=flat-square&logo=c%2B%2B)
![PlatformIO](https://img.shields.io/badge/PlatformIO-ESP32-F58220?style=flat-square&logo=platformio)
![AWS](https://img.shields.io/badge/AWS-IoT%20Core-232F3E?style=flat-square&logo=amazonaws)

기존 고정형 공기청정기의 사각지대 문제를 해결하기 위해, 라이다와 각종 센서, 경량 AI를 탑재하여 실내 공간을 자율주행하며 오염을 선제적으로 정화하는 가정용 이동형 공기청정 로봇입니다. (2026 한이음 ICT 멘토링 우수 프로젝트)

---

## System Overview

기존 고정형 공기청정기는 설치 위치 주변만 청정 가능하여 넓은 실내 공간이나 거실과 떨어진 방에는 사각지대가 발생합니다. ARIA는 센서를 본체에 직접 내장한 소형 가정용 구조로 설계되었으며, 시간대별 오염 발생 패턴을 학습하여 오염이 발생하기 전에 미리 이동하는 선제 대응 방식을 채택했습니다.

<!-- 로봇 외관 및 웹앱 화면 이미지 삽입 -->
<p align="center">

<img width="408" height="602" alt="Image" src="https://github.com/user-attachments/assets/43f0793d-f6d2-470c-a15d-644823c48bcc" />
<img width="255" height="591" alt="Image" src="https://github.com/user-attachments/assets/8600a1cb-af0b-49f9-ab8f-a2f128841ff5" />
</p>



## Core Architecture

단일 제어 루프의 병목을 극복하기 위해 **라즈베리파이(고성능 연산/AI)와 ESP32(실시간 하드웨어 제어)를 명확히 분리한 이기종 분산 아키텍처**로 설계되었습니다.

<!-- 첨부해주신 통합 아키텍처 이미지(image_1a1de5.png)의 깃허브 URL 삽입 -->
<p align="center">
  
<img width="1000" height="750" alt="Image" src="https://github.com/user-attachments/assets/b03f4e30-627a-4c1a-b035-aa65df5f5d68" />
</p>

### 1. Software Architecture (로봇-클라우드-웹 연동)
*   **로봇 내부 계층:** LiDAR, ToF, 공기질 센서(PM2.5, VOC, 온습도) 데이터를 수집하고, 실시간 지도 작성 및 자율주행 모듈(SLAM, Nav2)을 구동합니다.
*   **통신 / 클라우드 계층:** 로봇과 웹앱은 AWS IoT Core를 매개로 Device Shadow(desired/reported)를 통해 상태를 동기화합니다. 수신된 시계열 센서 데이터는 AWS Lambda를 거쳐 TimescaleDB의 Hypertable에 적재되어 빠른 조회를 지원합니다.
*   **웹 어플리케이션 (React):** QR 로그인, 실시간 위치 표시가 포함된 맵 시각화, 스케줄 설정, 원격 모드 제어 등 사용자 인터페이스를 제공합니다.

<p align="center">
  
<img width="650" height="750" alt="Image" src="https://github.com/user-attachments/assets/89c24b1c-c673-494a-bbcb-57c1cfec97cb" />
</p>

### 2. Hardware Architecture (이기종 분산 제어)
*   **메인보드 (Raspberry Pi 4B):** 자율주행 및 센서 데이터 분석, AI 판단 처리를 담당합니다.
*   **서브보드 (ESP32):** 각종 센서(공기질, 장애물 회피용 ToF, IMU)를 연결하여 데이터를 수집하고 구동부(바퀴, 모터)와 팬/필터를 실시간 통제합니다.
*   **UART 바이너리 패킷 통신:** ESP32와 라즈베리파이 간 통신은 데이터 크기를 최소화하고 파싱 속도를 높이기 위해 22Byte 바이너리 패킷 구조(`[Header 2B] [Packet ID 1B] [Payload] [Checksum 1B] [Tail 2B]`)로 직접 설계되었습니다.

---

## Key Technologies (핵심 적용 기술)

### 1. Edge AI 기반 오염 구역 선제 예측 (3단계 구조)
단발성 센서 반응에 의존하지 않고, 생활 오염원(요리, 환기 등)의 반복적인 시간 패턴을 학습합니다.
*   **1단계 (데이터 수집):** 로봇이 2시간 주기로 전 구역을 순회하며, 각 구역 도착 시점의 PM2.5 값을 시각, 요일 정보와 함께 CSV로 기록합니다.
*   **2단계 (야간 재학습):** 매일 새벽 3시, 누적된 데이터 중 실제 오염이 발생했던 시점(PM2.5 70 이상)의 데이터만 추출하여 Random Forest 분류기(결정트리 50개, 최대 깊이 10)를 재학습시킵니다.
*   **3단계 (실시간 예측 및 선제 이동):** 10초 주기로 현재 시각을 모델에 입력해 구역별 오염 확률을 산출하며, 예측 확률이 70%를 넘는 구역이 있으면 오염 확산 전 해당 구역으로 선제 이동하여 청정을 시작합니다.

### 2. ROS2 자율주행 및 공간 인식
*   **SLAM 및 위치 추정:** SLAM Toolbox(async 모드)의 루프 클로저(Loop Closure) 기법을 통해 2차원 지도를 생성하고, AMCL(Adaptive Monte Carlo Localization) 알고리즘으로 확률론적 위치 추정을 수행합니다.
*   **Watershed 구역 자동 분할:** 생성된 지도를 방, 복도, 주방 등으로 분할하기 위해 Distance Transform과 Watershed 알고리즘을 적용하여 Nav2의 이동 목표 지점을 자동 추출합니다.
*   **경로 계획 (Nav2):** A* 알고리즘 기반의 전역 경로 탐색과 DWB(Dynamic Window Approach) 기반의 지역 장애물 회피를 결합하여 주행합니다.

### 3. 절전 및 자원 최적화
*   **수면 시간 자동 절전:** 사용자가 설정한 수면 시간대(기본 23:00~07:00)에는 AI 추론을 자동으로 중단하는 SLEEP_MODE로 전환되어 불필요한 연산 자원 및 전력 소모를 방지합니다.

---

## Tech Stack

| Category | Technologies |
| :--- | :--- |
| **OS / Environment** | Ubuntu 22.04 LTS, ROS2 Humble, PlatformIO, Git/GitHub |
| **Hardware** | Raspberry Pi 4B, ESP32 Devkit C, Cytron 모터드라이버 |
| **Sensors** | YDLidar X2, VL53L0X(ToF), SGP40(공기질), HDC2080(온습도), MPU9250(IMU) |
| **Languages** | Python 3.10 (AI, ROS2), C++ (ESP32 펌웨어), JavaScript (웹앱) |
| **AI / Data** | scikit-learn (Random ForestClassifier) |
| **Cloud / Backend** | AWS IoT Core (MQTT), AWS Lambda, AWS S3, PostgreSQL (TimescaleDB) |
| **Communication** | UART 바이너리 통신, Wi-Fi, paho-mqtt |

---

## Expected Effects

*   **사각지대 없는 전 구역 공기청정:** 자율주행 기능을 통해 단 1대의 로봇이 실내 전 구역을 순회하므로 넓은 공간에서의 사각지대 문제를 해소합니다.
*   **진입 장벽 완화:** 외부 환경 센서를 별도로 설치해야 하는 기존 고가 제품들과 달리, 공기질 센서를 본체에 내장하여 단일 기기만으로 작동 가능하므로 초기 설치 부담이 낮습니다.
*   **확장성:** 청정 기능을 제외하고 공기질 센서만으로 다중 공간(급식실, 매장 등)의 대기를 측정하는 순찰 로봇으로도 활용 가치가 있습니다.

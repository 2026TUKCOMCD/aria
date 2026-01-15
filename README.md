# AI 로봇 공기청정기 (Aria)

## 1. 개요 (Overview)
본 프로젝트는 기존의 고정형 공기청정기가 가진 수동적 정화의 한계를 극복하고, 사용자의 생활 패턴을 학습하여 스스로 오염원을 찾아가는 시스템입니다. 

단순히 센서 수치가 높을 때 가동하는 사후 대응 방식을 넘어, AI 기반의 오염 조기 예측과 실시간 객체 인식(YOLO)을 통한 사람 활동 여부 판단을 결합하여 정화 효율과 에너지 절약이라는 두 가지 목표를 동시에 달성합니다. 

지능형 자율주행: SLAM(Simultaneous Localization and Mapping)과 Navigation2를 활용해 집안 지도를 생성하고 최적의 정화 위치로 이동합니다.
사용자 패턴 분석: 센서 데이터의 기울기(Gradient) 분석으로 요리 등의 이벤트를 예측하며, 사용자의 앱 설정 스케줄에 맞춰 저전력 모드와 정상 가동 모드를 스마트하게 전환합니다.
클라우드 데이터 파이프라인: AWS IoT Core와 Lambda를 연동하여 센서 데이터를 수집하고, 축적된 데이터를 통해 AI 모델을 지속적으로 고도화하는 MLOps 구조를 지향합니다.

---

## 2. 시스템 구조도 (System Architecture)
<img width="1141" height="592" alt="image" src="https://github.com/user-attachments/assets/6a8448ac-9c19-4d13-abdb-b5f5db7b0e16" />



---

## 3. 개발 환경 (Development Environment)


| 구분 | 상세 내용 |
| :--- | :--- |
| **OS** | Ubuntu 22.04 LTS |
| **Middleware** | ROS 2 Humble Hawksbill |
| **Language** | Python 3.10, C++ (Arduino IDE), JavaScript |
| **Cloud/DB** | AWS IoT Core, Lambda, PostgreSQL (TimescaleDB) |
| **AI/CV** | YOLOv8, PyTorch |


---

## 4. 운영 환경 (Operating Environment)


| 구분 | 상세 내용 |
| :--- | :--- |
| **Web** | Runs on Docker container on AWS EC2 |
| **Cloud** | S3, PostgreSQL, AI SageMaker, AWS IoT Core, Lambda |
| **Raspberry Pi 4B** | Programs using SLAM Toolbox and Nav2 run on ROS 2 middleware |
| **ESP32** | Code developed with Arduino runs on firmware |

---

## 5. 운영 시나리오 (Operating Scenario)
<img width="925" height="537" alt="image" src="https://github.com/user-attachments/assets/8553823b-e6ee-447c-851a-e122a8399d66" />

<img width="942" height="542" alt="image" src="https://github.com/user-attachments/assets/ddeb83ee-3d1a-4558-969e-d06f70800447" />

<img width="1040" height="737" alt="image" src="https://github.com/user-attachments/assets/e5c45bb8-4015-4fed-8dbc-4d03b58bf0f5" />

---

## 6. 데모 환경 (Demo Environment)

<img width="1218" height="557" alt="image" src="https://github.com/user-attachments/assets/ce715225-c9af-4bc0-ae4d-1b23c71ad247" />

---

## ☁️ Cloud & Data (IoT Backend) Feature & User Story Backlog(160h)

| ID | 유저스토리 (User Story) | SP | 시간 (h) |
|----|-------------------------|----|----------|
| A1-1 | 데이터 통신 규격 및 MQTT 토픽 설계 | 3 | 12 |
| A1-2 | AWS IoT Core 보안 연결 (TLS / SSL) | 5 | 20 |
| A1-3 | Keep-Alive 및 연결 상태 관리 | 4 | 16 |
| A2-1 | 센서 데이터 저장 파이프라인 구축 | 14 | 56 |
| A3-1 | Device Shadow JSON 스키마 정의 | 4 | 16 |
| A3-2 | 웹앱 명령(Desired)과 기기 상태(Reported) 동기화 구현 | 10 | 40 |

---

## 🤖 Autonomous Navigation (ROS2) Feature & User Story Backlog(588h)

| ID | 유저스토리 (User Story) | SP | 시간 (h) |
|----|-------------------------|----|----------|
| B0-1 | ROS2 워크스페이스 / 빌드 환경 구성 (colcon) | 2 | 8 |
| B0-2 | VSCode 개발환경 구성 | 2 | 8 |
| B0-3 | 시뮬레이션 실행 환경 준비 (RViz) | 3 | 12 |
| B0-4 | 로깅 / 디버깅 / 리플레이 (rosbag) 기반 구축 | 4 | 16 |
| B1-1 | URDF 작성 | 5 | 20 |
| B1-2 | TF 트리 구성 | 8 | 32 |
| B1-3 | `/odom` 계산 및 발행 | 10 | 40 |
| B1-4 | AMCL로 현재 위치 트래킹 및 튜닝 | 13 | 52 |
| B2-1 | LiDAR 드라이버 연동 | 5 | 20 |
| B2-2 | 2D 점유 격자 지도 생성 (SLAM) | 24 | 96 |
| B2-3 | 지도 저장 및 불러오기 기능 | 5 | 20 |
| B3-1 | 경로 생성 (Global Planner) | 8 | 32 |
| B3-2 | 실시간 로컬 경로 수정 (Local Planner) | 22 | 88 |
| B4-1 | 방 / 거실 영역 분할 | 13 | 52 |
| B4-2 | 최적 정화 위치 선정 | 10 | 40 |
| B4-3 | 공기질 취약 지역 우선 정화 경로 생성 | 13 | 52 |

---

## 🍳 Event AI (Cooking Detection) Feature & User Story Backlog(356h)

| ID | 유저스토리 (User Story) | SP | 시간 (h) |
|----|-------------------------|----|----------|
| C1-1 | 애매한 요리 이벤트 감지 시 부엌 이동 트리거 | 8 | 32 |
| C1-2 | 부엌 도착 직전 녹화 시작 | 5 | 20 |
| C1-3 | 부엌 도착 후 360도 회전 촬영 | 5 | 20 |
| C1-4 | 요리 이벤트 최종 판단 및 복귀 | 8 | 32 |
| C2-1 | 센서값 롤링 버퍼 형식 관리 | 5 | 20 |
| C2-2 | 조기 예측 트리거 발생 시 센서 데이터 전달 | 8 | 32 |
| C2-3 | 클라우드 기반 다음 이벤트 예측 | 8 | 32 |
| C3-1 | 요리 검증 결과 데이터화 및 자동 삭제 | 5 | 20 |
| C4-1 | 재학습용 요리 데이터 자동 수집 | 10 | 40 |
| C4-2 | 모델 재학습 (Fine-tuning) 자동화 | 14 | 56 |
| C5-1 | 요리 감지 AI 추론 엔진 및 라이브러리 설정 | 5 | 20 |
| C5-2 | AWS SageMaker / S3 초기 환경 세팅 | 5 | 20 |
| C6-1 | 요리 감지 모델 버전 관리 및 배포 정책 | 3 | 12 |

---

## ⚙️ Hardware & Device Feature Backlog(196h)

| ID | 유저스토리 (User Story) | SP | 시간 (h) |
|----|-------------------------|----|----------|
| D1-1 | 내부 센서 배치 및 설정 | 3 | 12 |
| D1-2 | 간섭 체크 및 발열 시뮬레이션 | 4 | 16 |
| D2-1 | 외형 구조물 3D 모델링 | 5 | 20 |
| D2-2 | 센서 시야 최적화 (카메라, 라이다) | 4 | 16 |
| D3-1 | 모터 PWM 제어 인터페이스 구현 | 4 | 16 |
| D3-2 | 센서 데이터 수집 인터페이스 구현 | 4 | 16 |
| D3-3 | 하드웨어 상태 모니터링 인터페이스 구현 | 4 | 16 |
| D3-4 | USB 시리얼 통신 규약 설계 | 8 | 32 |
| D3-5 | 엔코더 데이터를 읽어 바퀴의 회전수 계산 | 8 | 32 |
| D3-6 | 배터리 잔량 확인 및 안전 관리 | 5 | 20 |

---

## 📱 Web App Feature & User Story Backlog(288h)

| ID | 유저스토리 (User Story) | SP | 시간 (h) |
|----|-------------------------|----|----------|
| E1-1 | 라즈베리파이로부터 맵 데이터 수신 | 4 | 16 |
| E1-2 | 맵 데이터 요청 전송 (앱 → 라즈베리파이) | 3 | 12 |
| E1-3 | 맵 데이터 최신화 로직 | 4 | 16 |
| E1-4 | 맵 수신 실패 시 예외 처리 및 안내 | 2 | 8 |
| E1-5 | 맵 시각화 표시 및 로봇 실시간 위치 확인 | 10 | 40 |
| E1-6 | 맵 위 구역별 공기질 데이터 수신 및 갱신 | 5 | 20 |
| E1-7 | 공기질 상태를 색상으로 맵에 오버레이 표시 | 6 | 24 |
| E1-8 | 색상 범례 및 최신 갱신 시각 표시 | 3 | 12 |
| E1-9 | 데이터 누락/지연 시 표시 정책 (회색 처리 등) | 2 | 8 |
| E2-1 | On / Off로 AI 모드 제어 | 5 | 20 |
| E3-1~5 | 기본 기능 (라벨링, 상태 표시, On/Off 등) | 10 | 40 |
| E4-1 | AI 모드 작동 메뉴얼 표시 | 2 | 8 |
| E4-2 | 감지된 이벤트 종류 표시 | 4 | 16 |
| E5-1~3 | 오류 페이지 및 초기화 안내 | 2 | 8 |
| E6-1 | 사용자 취침 / 기상 스케줄 설정 | 10 | 40 |

---

## 📌 Feature & User Story Backlog (208h)

| ID   | 유저스토리 (User Story) | SP | 시간 (h) |
|------|-------------------------|----|----------|
| F1-1 | 구역별 360도 회전 촬영 및 활동 판단 | 7  | 28 |
| F2-1 | YOLO 기반 Active Person 판단 로직 | 8  | 32 |
| F2-2 | Sleeping / Lying Person 필터링 정책 적용 | 6  | 24 |
| F3-1 | 구역별 데이터 합산을 통한 최종 판단 | 5  | 20 |
| F4-1 | 저전력 모드 전환 및 웹 스케줄 우선 적용 | 5  | 20 |
| F4-2 | 재활성화 로직 | 3  | 12 |
| F5-1 | 라즈베리파이 추론 환경 구축 | 5  | 20 |
| F5-2 | Google Colab 학습 파이프라인 구축 | 10 | 40 |
| F6-1 | 모델 성능 모니터링 및 업데이트 | 3  | 12 |

### 합계: 360sp/1440h


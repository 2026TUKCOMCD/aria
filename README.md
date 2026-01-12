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
| **Language** | Python 3.10, C++ (Arduino IDE) |
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

## 클라우드 및 데이터 - 41sp(164h)
| ID  | User Story | SP | 예상 시간 |
|---|---|---:|---:|
| A1-1 | 데이터 통신 규격 및 MQTT 설계 | 4 | 16h |
| A1-2 | AWS IoT Core 보안 연결 (TLS) | 6 | 24h |
| A1-3 | Keep-Alive 및 연결 관리 | 3 | 12h |
| A2-1 | 센서 데이터 저장 파이프라인 구축<br>(IoT Rule -> Lambda -> Timescale) | 14 | 56h |
| A3-2 | Device Shadow 스키마 정의 | 4 | 16h |
| A3-2 | Shadow 기반 동기화 구현 | 10 | 40h 

---

## 자율 주행 - 126sp(504h)
| ID  | User Story | SP | 예상 시간 |
|---|---|---:|---:|
| B1-1 | URDF(로봇 모델링) 작성 | 4 | 16h |
| B1-2 | TF 트리 구성 | 6 | 24h |
| B1-3 | Odometry 계산 및 발행 | 14 | 56h |
| B1-4 | AMCL 위치 트래킹 | 10 | 40h |
| B2-1 | LiDAR 드라이버 연동 | 6 | 24h |
| B2-2 | 2D 점유 격자 지도 생성 | 24 | 96h |
| B2-3 | 지도 저장/불러오기 | 6 | 24h |
| B3-1 | Global Path Planning | 10 | 40h |
| B3-2 | Local Path Planning(회피) | 22 | 88h |
| B4-1 | 영역 분할(Segmentation) | 8 | 32h |
| B4-2 | 최적 청정 위치 선정 | 6 | 24h |
| B4-3 | 취약 지역 우선 경로 생성 | 10 | 40h |

---

## AI 상황 - 94sp(376h)
| ID  | User Story | SP | 예상 시간 |
|---|---|---:|---:|
| C1-1 | 부엌 이동 트리거 | 6 | 24h |
| C1-2 | 부엌 도착 직전 녹화 시작 | 4 | 16h |
| C1-3 | 부엌 도착 후 360도 회전 촬영 | 3 | 12h |
| C1-4 | 활동 중인 사람(Active Person)판단 | 10 | 40h |
| C1-5 | 요리 이벤트 최종 판단 및 복귀 | 6 | 24h |
| C2-1 | 구역별 촬영/판단 | 6 | 24h |
| C2-2 | YOLO 활동 객체 필터링 정책 | 6 | 24h |
| C2-3 | 구역별 합산 최종 비활동 판단 | 4 | 16h |
| C2-4 | 저전력 모드 및 센서 비활성화 | 8 | 32h |
| C3-1 | 영상 처리 결과 메타데이터화 및 자동 삭제 | 4 | 16h |
| C4-1 | 센서 드롭 버퍼 관리 | 4 | 16h |
| C4-2 | 조기 예측 트리거 센서 데이터 전달 | 6 | 24h |
| C4-3 | 클라우드 기반 다음 이벤트 예측 | 8 | 32h |
| C5-1 | 재학습용 데이터 자동 수집 | 6 | 24h |
| C5-2 | 모델 재학습 자동화 | 8 | 32h |
| C5-3 | 모델 검증/버전 관리 | 5 | 20h |

---

## 하드웨어 - 55sp(220h)
| ID  | User Story | SP | 예상 시간 |
|---|---|---:|---:|
| D1-1 | 내부 센서 및 부품 배치 | 4 | 16h |
| D1-2 | 발열 및 간섭 시뮬레이션 | 4 | 16h |
| D2-1 | 외형 구조물 3D 모델링 | 5 | 24h |
| D2-2 | 센서 시야(FOV) 최적화 | 6 | 24h |
| D3-1 | 모터 PWM제어 인터페이스 | 8 | 32h |
| D3-2 | 센서 데이터 수집 인터페이스 | 6 | 24h |
| D3-3 | 하드웨어 상태 모니터링 | 5 | 20h |
| D3-4 | USB 시리얼 통신 구성 설계 | 4 | 16h |
| D3-5 | 엔코더 회전수 계산 | 8 | 32h |
| D3-6 | 배터리 모니터링 및 안전 관리 | 4 | 16h |

---

## Web App - 44sp(176h)
| ID  | User Story | SP | 예상 시간 |
|---|---|---:|---:|
| E1-1 | 맵 데이터 수신 | 2 | 8h |
| E1-2 | 맵 데이터 갱신 요청 | 2 | 8h |
| E1-3 | 맵 최신화 및 로딩 UI | 2 | 8h |
| E1-4 | 수신 실패 예외 처리 | 2 | 8h |
| E1-5 | 맵 렌더링 로봇 위치 | 4 | 16h |
| E1-6 | 구역별 공기질 데이터 수신 | 2 | 8h |
| E1-7 | 공기질 색상 오버레이 | 2 | 8h |
| E1-8 | 범례 및 갱신 시각 표시 | 2 | 8h |
| E1-9 | 데이터 누락 처리 | 2 | 8h |
| E2-1 | AI 모드 on/off | 2 | 8h |
| E3-1 | 라벨링 설정 | 2 | 8h |
| E3-2 | 현재 상태 표시 | 3 | 12h |
| E3-3 | 공기청정기 on/off | 2 | 8h |
| E3-4 | 공기청정기 팬풍 | 3 | 12h |
| E3-5 | 공기청정기 위치 설정 | 3 | 12h |
| E4-1 | AI 모드 버튼열 표시 | 2 | 8h |
| E4-2 | 감지 이벤트 표시 | 2 | 8h |
| E5-1 | 통신 오류 페이지 | 1 | 4h |
| E5-2 | 시스템 오류 페이지 | 1 | 4h |
| E5-3 | 오류 후 초기화 안내 | 1 | 4h |
| E6-1 | 취침/기상 스케줄 설정 | 3 | 12h |


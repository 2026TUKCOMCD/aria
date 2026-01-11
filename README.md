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
<img width="925" height="537" alt="image" src="https://github.com/user-attachments/assets/8553823b-e6ee-447c-851a-e122a8399d66" />

<img width="942" height="542" alt="image" src="https://github.com/user-attachments/assets/ddeb83ee-3d1a-4558-969e-d06f70800447" />

<img width="1040" height="737" alt="image" src="https://github.com/user-attachments/assets/e5c45bb8-4015-4fed-8dbc-4d03b58bf0f5" />

---

## 5. 데모 환경 (Demo Environment)

<img width="1118" height="457" alt="image" src="https://github.com/user-attachments/assets/ce715225-c9af-4bc0-ae4d-1b23c71ad247" />

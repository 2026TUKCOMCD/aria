# AWS IoT Core Secure Connection Guide (A1-2)

## 1. Prerequisites
이 코드를 실행하기 위해서는 AWS IoT Core에서 발급받은 인증서 파일이 필요합니다.
보안상의 이유로 인증서는 GitHub에 포함되지 않으며, `.gitignore` 처리되어 있습니다.

### 1.1 Hardware Requirement
- **Device**: Raspberry Pi 4B (Main) or MacBook (Testing)
- **OS**: Raspberry Pi OS (64-bit recommended)
- **Language**: Python 3.10+

### 1.2 Setup Certificates
프로젝트 루트에 `certs` 폴더를 생성하고, 인증서 파일 3개를 위치시키세요.
(파일명은 `scripts/aws_connect_test.py` 코드 내의 경로와 일치해야 합니다.)

```text
aria/
├── certs/ (Git Ignored)
│   ├── AmazonRootCA1.pem
│   ├── certificate.pem.crt    <-- (파일명 확인 필수)
│   └── private.pem.key        <-- (파일명 확인 필수)
├── scripts/
│   └── aws_connect_test.py
└── ...

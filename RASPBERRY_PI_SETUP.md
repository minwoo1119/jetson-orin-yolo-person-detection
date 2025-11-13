# Raspberry Pi Setup Guide

이 프로젝트를 라즈베리파이에서 실행하기 위한 가이드입니다.

## 시스템 요구사항

- **Raspberry Pi 4 또는 5** (최소 4GB RAM 권장)
- **Raspberry Pi OS** (64-bit 권장)
- **Python 3.8+**

## 설치 단계

### 1. 시스템 업데이트

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 필수 라이브러리 설치

```bash
# 시스템 라이브러리
sudo apt install -y python3-pip python3-opencv libopencv-dev
sudo apt install -y libssl-dev libffi-dev

# Python 패키지
pip3 install --upgrade pip
pip3 install psutil pyyaml pandas matplotlib seaborn numpy
pip3 install opencv-python-headless  # 라즈베리파이용 경량 버전
```

### 3. PyTorch 설치 (CPU 버전)

```bash
# CPU 전용 PyTorch 설치 (CUDA 불필요)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. YOLO (Ultralytics) 설치

```bash
pip3 install ultralytics
```

### 5. 프로토콜 라이브러리 설치

```bash
# HTTPS/TLS (기본 제공)
# MQTT
pip3 install paho-mqtt

# HTTP/3 QUIC
pip3 install aioquic

# CoAP
pip3 install aiocoap

# DTLS
pip3 install pyopenssl
```

## 설정 파일 수정

라즈베리파이에 맞게 설정을 조정하세요:

```yaml
# configs/suite.yaml
board: "raspberry_pi"  # ← 변경
num_models: 1          # ← 4에서 1로 줄이기 (메모리 절약)
duration: 30           # ← 60에서 30으로 줄이기 (선택사항)
warmup: 3              # ← 5에서 3으로 줄이기 (선택사항)

# 패킷 크기와 전송률도 낮추는 것을 권장
payload_sizes: [1024, 4096]      # 대신 [4096, 16384, 65536]
send_rates: [50, 100]            # 대신 [100, 500, 1000]
```

## 성능 최적화 팁

### 1. 경량 YOLO 모델 사용

```python
# yolo11n.pt 대신 더 작은 모델 사용 (선택사항)
# configs/suite.yaml
model_path: "yolo_model/yolo11n.pt"  # nano 모델 (가장 빠름)
```

### 2. 해상도 낮추기

비디오 해상도를 낮춰서 처리 속도 향상:

```python
# yolo_runner.py에서 프레임 리사이즈 (필요시)
frame = cv2.resize(frame, (640, 480))  # 또는 (416, 416)
```

### 3. 스왑 메모리 증가 (메모리 부족 시)

```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048 로 변경 (기본 100MB → 2GB)
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 실행 방법

### 1. 테스트 실행

```bash
# 작은 테스트부터 시작
python3 run_suite.py --config configs/suite.yaml
```

### 2. 결과 확인

```bash
# 실험 완료 후 출력되는 디렉토리 확인
ls results/

# 시각화
python3 visualize_results.py --result-dir results/YYYYMMDD_HHMMSS_label
```

## 예상 성능

| 항목 | Jetson Orin Nano | Raspberry Pi 4 | Raspberry Pi 5 |
|------|------------------|----------------|----------------|
| YOLO FPS | 30-60 | 3-8 | 8-15 |
| 메모리 사용 | ~2GB | ~1.5GB | ~1.5GB |
| 프로토콜 처리 | 실시간 | 실시간 | 실시간 |
| 권장 모델 수 | 4개 | 1개 | 1-2개 |

## 문제 해결

### GPU 관련 경고 무시

다음 경고는 정상입니다 (CPU 모드에서):
```
Warning: aiocoap.transports.dtls not available, using PyOpenSSL DTLS implementation
Using CPU
```

### 메모리 부족 에러

```bash
# num_models를 1로 줄이거나 스왑 메모리 증가
# 또는 더 작은 YOLO 모델 사용
```

### 느린 실행 속도

라즈베리파이는 Jetson보다 느립니다:
- `duration` 줄이기 (30초로)
- `num_models: 1` 사용
- `send_rates` 낮추기 ([50, 100])

## 라즈베리파이 vs Jetson 차이점

| 기능 | Jetson | Raspberry Pi |
|------|--------|--------------|
| GPU 가속 | ✅ CUDA | ❌ CPU만 |
| jtop 모니터링 | ✅ | ❌ (psutil만) |
| YOLO 속도 | 빠름 | 느림 |
| 프로토콜 테스트 | ✅ | ✅ (동일) |
| RTT 측정 | ✅ | ✅ (동일) |

## 추가 문제가 있다면

1. Python 버전 확인: `python3 --version` (3.8+ 필요)
2. 메모리 확인: `free -h`
3. 로그 확인: 실행 중 출력되는 에러 메시지 확인

import cv2
from ultralytics import YOLO
import time
import os
import psutil
import torch

# GPU 관련 라이브러리는 try-except로 처리하여 GPU 없는 환경에서도 동작하도록 함
try:
    import pynvml
    HAS_GPU = torch.cuda.is_available()
    if not HAS_GPU:
        print("NVIDIA GPU가 감지되었지만, CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
except ImportError:
    HAS_GPU = False
    print("pynvml 라이브러리가 설치되지 않았거나, NVIDIA GPU 환경이 아닙니다. CPU를 사용합니다.")
    
# ----------------------------------------------------
# 1. 모델 및 비디오 경로 설정
# ----------------------------------------------------

try:
    model_path = os.path.join('yolo_model', 'yolo11n.pt')
    model = YOLO(model_path)
    print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    print("yolo_model/yolov8n.pt 파일이 올바른 위치에 있는지 확인해주세요.")
    exit()

video_files = {
    # "Minimal": os.path.join('videos', 'minimal.mp4'),
    "Typical": os.path.join('videos', 'KITTI-19-raw.webm'),
    # "Stress": os.path.join('videos', 'stress.mp4')
}

# ----------------------------------------------------
# 2. 시스템 모니터링 초기화
# ----------------------------------------------------

gpu_handle = None
if HAS_GPU:
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"GPU 모니터링 초기화 실패: {e}")
        gpu_handle = None

# ----------------------------------------------------
# 3. 객체 탐지 함수 정의
# ----------------------------------------------------

def run_person_detection(video_path, video_type):
    """
    주어진 비디오 파일에 대해 'person' 객체 탐지를 수행하고 FPS와 시스템 자원 사용량을 측정합니다.
    """
    if not os.path.exists(video_path):
        print(f"오류: '{video_path}' 파일이 존재하지 않습니다.")
        return

    print(f"\n--- {video_type} 비디오 탐지 시작 ---")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"오류: 비디오 파일 '{video_path}'을 열 수 없습니다.")
        return

    frame_count = 0
    start_time = time.time()
    total_fps = 0
    
    target_classes = [0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # GPU가 있는 경우 'cuda' 사용, 없으면 'cpu' 사용
        device_arg = 'cuda' if HAS_GPU else 'cpu'
        
        # YOLO 모델로 객체 탐지 수행
        results = model.predict(source=frame, classes=target_classes, verbose=False, device=device_arg)
        
        # 결과 시각화
        annotated_frame = results[0].plot()
        
        # 시스템 정보 가져오기
        cpu_percent = psutil.cpu_percent()
        cpu_mem = psutil.virtual_memory()
        cpu_mem_percent = cpu_mem.percent

        gpu_percent, gpu_mem_percent = 0, 0
        if gpu_handle:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                gpu_percent = gpu_util.gpu
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                gpu_mem_percent = (gpu_mem_info.used / gpu_mem_info.total) * 100
            except pynvml.NVMLError as err:
                pass
        
        # FPS 및 시스템 정보 텍스트로 추가
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            
            # 텍스트와 배경을 위한 변수
            text_color = (255, 255, 255) # 흰색 텍스트
            bg_color = (0, 0, 0) # 검은색 배경
            alpha = 0.5 # 배경 투명도 (0.0은 투명, 1.0은 불투명)
            
            # FPS, CPU, GPU 정보를 담을 문자열 리스트
            info_texts = [
                f"FPS: {fps:.2f}",
                f"CPU: {cpu_percent:.1f}%",
                f"RAM: {cpu_mem_percent:.1f}%"
            ]
            if HAS_GPU:
                info_texts.append(f"GPU: {gpu_percent:.1f}%")
                info_texts.append(f"GPU Mem: {gpu_mem_percent:.1f}%")
            
            # 오버레이 이미지 생성
            overlay = annotated_frame.copy()
            
            # 배경 사각형 그리기
            # 필요한 경우 좌표와 사각형 크기 조정
            box_height = 40 * len(info_texts) + 10
            cv2.rectangle(overlay, (0, 0), (280, box_height), bg_color, -1)
            
            # 원본 프레임과 오버레이를 합성
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
            # 텍스트 그리기
            y_pos = 40
            for text in info_texts:
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                y_pos += 40
            
            # 터미널에 로그로 출력
            print(f"FPS: {fps:.2f} | CPU: {cpu_percent:.1f}% | RAM: {cpu_mem_percent:.1f}%", end="")
            if HAS_GPU:
                print(f" | GPU: {gpu_percent:.1f}% | GPU Mem: {gpu_mem_percent:.1f}%")
            else:
                print() # 줄 바꿈
            
            total_fps += fps
            frame_count = 0
            start_time = time.time()
            
        cv2.imshow(f"{video_type} Person Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 평균 FPS 계산
    avg_fps = total_fps / (time.time() - start_time)
    print(f"\n--- {video_type} 비디오 평균 FPS: {avg_fps:.2f} ---")
    
    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------
# 4. 메인 실행 루프
# ----------------------------------------------------

if __name__ == "__main__":
    for video_type, video_path in video_files.items():
        run_person_detection(video_path, video_type)

    if HAS_GPU and gpu_handle:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as err:
            pass

    print("\n모든 비디오 탐지 테스트가 완료되었습니다.")
    print("\n모든 비디오 탐지 테스트가 완료되었습니다.")
    print("\n모든 비디오 탐지 테스트가 완료되었습니다.")
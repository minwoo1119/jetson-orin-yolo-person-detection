import cv2
from ultralytics import YOLO
import time
import os

# ----------------------------------------------------
# 1. 모델 및 비디오 경로 설정
# ----------------------------------------------------

# YOLOv8 모델 로드
# 모델 파일 경로를 yolo_model 디렉토리 안의 yolov11n.pt로 지정
try:
    model_path = os.path.join('yolo_model', 'yolov11n.pt')
    model = YOLO(model_path)
    print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    print("yolo_model/yolov11ㄴn.pt 파일이 올바른 위치에 있는지 확인해주세요.")
    exit()

# 테스트 비디오 파일 목록
video_files = {
    "Minimal": os.path.join('videos', 'minimal.mp4'),
    "Typical": os.path.join('videos', 'typical.mp4'),
    "Stress": os.path.join('videos', 'stress.mp4')
}

# ----------------------------------------------------
# 2. 객체 탐지 함수 정의
# ----------------------------------------------------

def run_person_detection(video_path, video_type):
    """
    주어진 비디오 파일에 대해 'person' 객체 탐지를 수행하고 FPS를 측정합니다.
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
    
    # 클래스 ID 0 (Person)만 탐지하도록 설정
    # COCO 데이터셋에서 'person' 클래스 ID는 0
    target_classes = [0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO 모델로 객체 탐지 수행
        # classes=0 옵션으로 'person' 클래스만 탐지
        results = model.predict(source=frame, classes=target_classes, verbose=False, device='cuda')
        
        # 결과 시각화
        annotated_frame = results[0].plot()
        
        # 프레임 카운트 및 FPS 계산
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # 5초마다 FPS 출력
        if elapsed_time > 5:
            fps = frame_count / elapsed_time
            print(f"현재 FPS: {fps:.2f}")
            total_fps += fps
            frame_count = 0
            start_time = time.time()
            
        cv2.imshow(f"{video_type} Person Detection", annotated_frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 비디오 재생이 끝나면 평균 FPS 계산
    avg_fps = total_fps / (len(results))
    print(f"\n--- {video_type} 비디오 평균 FPS: {avg_fps:.2f} ---")
    
    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------
# 3. 메인 실행 루프
# ----------------------------------------------------

if __name__ == "__main__":
    # 각 비디오에 대해 반복적으로 탐지 함수 실행
    for video_type, video_path in video_files.items():
        run_person_detection(video_path, video_type)

    print("\n모든 비디오 탐지 테스트가 완료되었습니다.")
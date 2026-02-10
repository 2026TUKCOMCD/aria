from ultralytics import YOLO
import numpy as np
import time

# ==================== 키포인트 정의 ====================
# COCO 포맷 17개 키포인트 (YOLOv8-pose 기준)
KEYPOINT_NAMES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

# 필터링에 사용할 주요 키포인트 그룹
KEY_INDICES = {
    'shoulders': [5, 6],      # 좌/우 어깨
    'hips': [11, 12],         # 좌/우 골반
    'knees': [13, 14]         # 좌/우 무릎
}

# 주요 키포인트를 하나의 리스트로 (중복 제거)
IMPORTANT_KEYPOINTS = list(set(
    KEY_INDICES['shoulders'] + 
    KEY_INDICES['hips'] + 
    KEY_INDICES['knees']
))  # [5, 6, 11, 12, 13, 14]


class PersonDetector:
    """
    YOLOv8-pose 기반 사람 검출 클래스
    
    주요 기능:
    - 이미지/numpy array에서 사람 검출
    - 17개 키포인트 추출
    - 신뢰도 기반 필터링
    """
    
    def __init__(self, model_path='yolov8n-pose.pt', verbose=True):
        """
        Args:
            model_path: YOLO 모델 경로
            verbose: 로그 출력 여부
        """
        self.verbose = verbose
        
        if self.verbose:
            print(f"[INFO] Loading model: {model_path}")
        
        self.model = YOLO(model_path)
        
        if self.verbose:
            print("[INFO] Model loaded successfully!")
        
    def detect_persons(self, image_source):
        """
        이미지에서 사람 검출 및 키포인트 추출
        
        Args:
            image_source: 이미지 파일 경로(str) 또는 numpy array (OpenCV 프레임)
            
        Returns:
            list: 검출된 사람 정보 리스트
            [
                {
                    'bbox': [x1, y1, x2, y2],    # 좌상단, 우하단 좌표
                    'confidence': float,          # bbox 신뢰도
                    'keypoints': [[x, y, conf], ...],  # 17개
                    'num_keypoints': int,         # 키포인트 개수
                    'avg_keypoint_conf': float    # 주요 키포인트 평균 신뢰도
                }
            ]
        """
        start_time = time.time()
        
        # 입력 타입 확인
        if isinstance(image_source, str):
            if self.verbose:
                print(f"\n[INFO] Detecting persons in: {image_source}")
        elif isinstance(image_source, np.ndarray):
            if self.verbose:
                print(f"\n[INFO] Detecting persons in numpy array (shape: {image_source.shape})")
        else:
            raise TypeError(f"image_source must be str or numpy.ndarray, got {type(image_source)}")
        
        # YOLO 추론
        results = self.model(image_source, verbose=False)
        
        persons = []
        
        for result in results:
            # 검출된 객체가 없으면 스킵
            if result.boxes is None or len(result.boxes) == 0:
                if self.verbose:
                    print("[WARN] No objects detected")
                continue
                
            # 키포인트가 없으면 스킵
            if result.keypoints is None or len(result.keypoints.data) == 0:
                if self.verbose:
                    print("[WARN] No keypoints detected")
                continue
            
            # 각 사람 처리
            for idx, (box, keypoints) in enumerate(zip(result.boxes.data, result.keypoints.data)):
                # Bounding box: [x1, y1, x2, y2, confidence, class]
                bbox = box[:4].cpu().numpy()  # x1, y1, x2, y2
                confidence = float(box[4])
                
                # 키포인트: [17, 3] (x, y, confidence)
                kps = keypoints.cpu().numpy()
                
                # 주요 키포인트의 평균 신뢰도 계산
                key_confidences = kps[IMPORTANT_KEYPOINTS, 2]  # confidence 값만
                avg_kp_conf = float(np.mean(key_confidences))
                
                person_info = {
                    'bbox': bbox.tolist(),
                    'confidence': confidence,
                    'keypoints': kps.tolist(),
                    'num_keypoints': len(kps),
                    'avg_keypoint_conf': avg_kp_conf
                }
                
                persons.append(person_info)
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"[INFO] Detected {len(persons)} person(s) in {elapsed:.3f}s")
            for idx, person in enumerate(persons):
                print(f"  Person {idx+1}: bbox_conf={person['confidence']:.2f}, "
                      f"kp_conf={person['avg_keypoint_conf']:.2f}")
        
        return persons
    
    def filter_by_confidence(self, persons, min_bbox_conf=0.5, min_keypoint_conf=0.5):
        """
        신뢰도 기준으로 필터링
        
        Args:
            persons: detect_persons() 결과
            min_bbox_conf: 최소 bounding box 신뢰도 (0.0 ~ 1.0)
            min_keypoint_conf: 최소 키포인트 평균 신뢰도 (0.0 ~ 1.0)
            
        Returns:
            list: 필터링된 사람 리스트
        """
        filtered = []
        
        for person in persons:
            # Bounding box 신뢰도 체크
            if person['confidence'] < min_bbox_conf:
                continue
            
            # 키포인트 평균 신뢰도 체크 (이미 계산되어 있음!)
            if person['avg_keypoint_conf'] < min_keypoint_conf:
                continue
            
            filtered.append(person)
        
        if self.verbose:
            print(f"\n[INFO] Filtered: {len(filtered)}/{len(persons)} persons")
            print(f"  (bbox_conf>={min_bbox_conf}, kp_conf>={min_keypoint_conf})")
        
        return filtered
    
    def get_keypoint_by_name(self, person, keypoint_name):
        """
        키포인트 이름으로 좌표 가져오기
        
        Args:
            person: detect_persons() 결과의 한 항목
            keypoint_name: 키포인트 이름 (예: 'nose', 'left_shoulder')
            
        Returns:
            tuple: (x, y, confidence) 또는 None (존재하지 않으면)
            
        Example:
            >>> nose = detector.get_keypoint_by_name(person, 'nose')
            >>> print(nose)  # (320.5, 180.2, 0.95)
        """
        # 이름으로 인덱스 찾기
        kp_index = None
        for idx, name in KEYPOINT_NAMES.items():
            if name == keypoint_name:
                kp_index = idx
                break
        
        if kp_index is None:
            if self.verbose:
                print(f"[WARN] Unknown keypoint name: {keypoint_name}")
            return None
        
        keypoints = person['keypoints']
        if kp_index >= len(keypoints):
            return None
        
        return tuple(keypoints[kp_index])  # [x, y, conf]


# ==================== 테스트 코드 ====================
if __name__ == '__main__':
    print("="*60)
    print("PersonDetector Test (Final Version)")
    print("="*60)
    
    # 검출기 생성
    detector = PersonDetector(verbose=True)
    
    # 테스트 1: 이미지 파일로 검출
    print("\n[TEST 1] Image file detection")
    persons = detector.detect_persons('test_images/bus.jpg')
    
    # 테스트 2: 신뢰도 필터링
    print("\n[TEST 2] Confidence filtering")
    filtered = detector.filter_by_confidence(
        persons, 
        min_bbox_conf=0.5, 
        min_keypoint_conf=0.5
    )
    
    # 테스트 3: 특정 키포인트 가져오기
    if len(filtered) > 0:
        print("\n[TEST 3] Get specific keypoint")
        person = filtered[0]
        nose = detector.get_keypoint_by_name(person, 'nose')
        left_shoulder = detector.get_keypoint_by_name(person, 'left_shoulder')
        
        if nose:
            print(f"  Nose: x={nose[0]:.1f}, y={nose[1]:.1f}, conf={nose[2]:.2f}")
        if left_shoulder:
            print(f"  Left Shoulder: x={left_shoulder[0]:.1f}, y={left_shoulder[1]:.1f}, conf={left_shoulder[2]:.2f}")
    
    print("\n" + "="*60)
    print(f"Test completed! Found {len(filtered)} high-confidence person(s)")
    print("="*60)

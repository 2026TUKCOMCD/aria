from activity_ai.person_detector import PersonDetector, KEYPOINT_NAMES
import numpy as np

class PoseClassifier:
    """
    키포인트 기반 자세 분류기 (비율 기반 - 개선됨!)
    
    지원 자세:
    - standing: 서있음
    - sitting: 앉아있음
    - lying: 누워있음
    """
    
    def __init__(self, lying_threshold=50, sitting_threshold=400, verbose=True):
        """
        Args:
            lying_threshold: 누움 판단 기준 (폴백용, 사용 안 할 수도 있음)
            sitting_threshold: 앉음 판단 기준 (픽셀)
            verbose: 로그 출력 여부
        """
        self.lying_threshold = lying_threshold
        self.sitting_threshold = sitting_threshold
        self.verbose = verbose
        
        if self.verbose:
            print(f"[INFO] PoseClassifier initialized (ratio-based)")
            print(f"  - lying_threshold: {lying_threshold}px (fallback)")
            print(f"  - sitting_threshold: {sitting_threshold}px")
    
    def _get_keypoint(self, person, keypoint_name):
        """
        키포인트 이름으로 좌표 가져오기 (내부 헬퍼 함수)
        
        Args:
            person: detect_persons() 결과의 한 항목
            keypoint_name: 키포인트 이름
            
        Returns:
            tuple: (x, y, confidence) 또는 None
        """
        # 이름으로 인덱스 찾기
        kp_index = None
        for idx, name in KEYPOINT_NAMES.items():
            if name == keypoint_name:
                kp_index = idx
                break
        
        if kp_index is None:
            return None
        
        keypoints = person['keypoints']
        if kp_index >= len(keypoints):
            return None
        
        kp = keypoints[kp_index]
        
        # 신뢰도가 너무 낮으면 None 반환
        if kp[2] < 0.3:  # confidence < 0.3
            return None
        
        return tuple(kp)  # (x, y, conf)
    
    def is_lying_down(self, person):
        """
        누워있는지 판단 (상체/하체 비율 방식)
        
        판단 로직:
        - 상체 길이 / 하체 길이 비율 사용
        - 누우면 상체 ≈ 하체 (ratio ≈ 1.0)
        - 서있으면 상체 < 하체 (ratio < 0.8)
        - 앉으면 상체 > 하체 (ratio > 1.5)
        
        Args:
            person: detect_persons() 결과의 한 항목
            
        Returns:
            bool: 누워있으면 True
        """
        # 필요한 키포인트 가져오기
        nose = self._get_keypoint(person, 'nose')
        left_hip = self._get_keypoint(person, 'left_hip')
        right_hip = self._get_keypoint(person, 'right_hip')
        left_ankle = self._get_keypoint(person, 'left_ankle')
        right_ankle = self._get_keypoint(person, 'right_ankle')
        
        # 키포인트가 없으면 판단 불가
        if not all([nose, left_hip, right_hip]):
            if self.verbose:
                print("  [WARN] Missing keypoints for lying detection")
            return False
        
        # 골반 중심점 계산
        hip_y = (left_hip[1] + right_hip[1]) / 2
        
        # 상체 길이
        upper_body = abs(hip_y - nose[1])
        
        # 발목이 있으면 하체 길이 계산
        if left_ankle or right_ankle:
            # 발목 중심점
            if left_ankle and right_ankle:
                ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            elif left_ankle:
                ankle_y = left_ankle[1]
            else:
                ankle_y = right_ankle[1]
            
            # 하체 길이
            lower_body = abs(ankle_y - hip_y)
            
            # 0으로 나누기 방지
            if lower_body > 10:
                # 상체 / 하체 비율
                ratio = upper_body / lower_body
                
                # 누움 판단: 상체와 하체가 비슷하면 누워있음
                # ratio가 0.7~1.3 범위면 누워있다고 판단
                is_lying = ratio > 1.2
                
                if self.verbose:
                    print(f"  [lying] upper={upper_body:.1f}, lower={lower_body:.1f}, ratio={ratio:.2f}")
                    print(f"  [lying] → {'LYING' if is_lying else 'not lying'} (0.7 < ratio < 1.3)")
                
                return is_lying
        
        # 발목이 없으면 절대값으로 폴백
        is_lying = upper_body < self.lying_threshold
        
        if self.verbose:
            print(f"  [lying] upper={upper_body:.1f}, threshold={self.lying_threshold}")
            print(f"  [lying] → {'LYING' if is_lying else 'not lying'} (fallback)")
        
        return is_lying
    
    def is_sitting(self, person):
        """
        앉아있는지 판단
        
        판단 로직:
        - 무릎이 골반보다 높거나 비슷한 높이면 앉아있음
        - 서있으면 무릎이 골반보다 훨씬 아래
        
        Args:
            person: detect_persons() 결과의 한 항목
            
        Returns:
            bool: 앉아있으면 True
        """
        # TODO: 쪼그려 앉은 경우 처리 (무릎-발목 각도 계산)
        # TODO: 의자에 앉은 것 vs 바닥에 앉은 것 구분
        
        # 필요한 키포인트 가져오기
        left_hip = self._get_keypoint(person, 'left_hip')
        right_hip = self._get_keypoint(person, 'right_hip')
        left_knee = self._get_keypoint(person, 'left_knee')
        right_knee = self._get_keypoint(person, 'right_knee')
        
        # 키포인트가 없으면 판단 불가
        if not all([left_hip, right_hip]):
            if self.verbose:
                print("  [WARN] Missing hip keypoints for sitting detection")
            return False
        
        # 골반 중심점
        hip_y = (left_hip[1] + right_hip[1]) / 2
        
        # 무릎이 하나라도 있으면 확인
        knee_ys = []
        if left_knee:
            knee_ys.append(left_knee[1])
        if right_knee:
            knee_ys.append(right_knee[1])
        
        if not knee_ys:
            if self.verbose:
                print("  [WARN] Missing knee keypoints for sitting detection")
            return False
        
        # 무릎 평균 y좌표
        knee_y = np.mean(knee_ys)
        
        # 무릎이 골반보다 위에 있거나 비슷하면 앉아있음
        # sitting_threshold 만큼 여유를 줌 (완전히 같지 않아도 OK)
        is_sitting = (knee_y - hip_y) < self.sitting_threshold
        
        if self.verbose:
            print(f"  [sitting] hip_y={hip_y:.1f}, knee_y={knee_y:.1f}, "
                  f"diff={knee_y - hip_y:.1f}, threshold={self.sitting_threshold} "
                  f"→ {'SITTING' if is_sitting else 'not sitting'}")
        
        return is_sitting
    
    def is_standing(self, person):
        """
        서있는지 판단
        
        판단 로직:
        - 누워있지도 않고, 앉아있지도 않으면 서있음
        
        Args:
            person: detect_persons() 결과의 한 항목
            
        Returns:
            bool: 서있으면 True
        """
        # FIXME: 애매한 각도 처리 개선 필요 (45도 기울어진 경우 등)
        
        lying = self.is_lying_down(person)
        sitting = self.is_sitting(person)
        
        # 누워있지도 않고, 앉아있지도 않으면 서있음
        is_standing = not lying and not sitting
        
        if self.verbose:
            print(f"  [standing] lying={lying}, sitting={sitting} "
                  f"→ {'STANDING' if is_standing else 'not standing'}")
        
        return is_standing
    
    def classify_pose(self, person):
        """
        자세 분류 (통합 함수)
        
        Args:
            person: detect_persons() 결과의 한 항목
            
        Returns:
            str: "lying", "sitting", "standing" 중 하나
        """
        if self.verbose:
            print(f"\n[INFO] Classifying pose...")
        
        # 우선순위: 누움 > 앉음 > 서있음
        # (누워있으면 일단 누움으로 판단)
        
        if self.is_lying_down(person):
            pose = "lying"
        elif self.is_sitting(person):
            pose = "sitting"
        else:
            pose = "standing"
        
        if self.verbose:
            print(f"[INFO] Final pose: {pose.upper()}")
        
        return pose
    
    def classify_all(self, persons):
        """
        여러 사람의 자세를 한번에 분류
        
        Args:
            persons: detect_persons() 결과 리스트
            
        Returns:
            list: 각 사람의 자세 리스트 ["standing", "sitting", ...]
        """
        poses = []
        
        for idx, person in enumerate(persons):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Person {idx + 1}:")
                print('='*60)
            
            pose = self.classify_pose(person)
            poses.append(pose)
        
        return poses


# ==================== 테스트 코드 ====================
if __name__ == '__main__':
    print("="*60)
    print("PoseClassifier Test (Ratio-based)")
    print("="*60)
    
    # 1. 사람 검출
    print("\n[STEP 1] Detecting persons...")
    detector = PersonDetector(verbose=True)
    persons = detector.detect_persons('test_images/bus.jpg')
    
    # 2. 신뢰도 필터링
    print("\n[STEP 2] Filtering by confidence...")
    filtered = detector.filter_by_confidence(persons, min_bbox_conf=0.5, min_keypoint_conf=0.5)
    
    # 3. 자세 분류
    print("\n[STEP 3] Classifying poses...")
    classifier = PoseClassifier(
        lying_threshold=50,
        sitting_threshold=400,
        verbose=True
    )
    
    poses = classifier.classify_all(filtered)
    
    # 4. 결과 요약
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for idx, (person, pose) in enumerate(zip(filtered, poses)):
        print(f"Person {idx + 1}: {pose.upper()} (confidence: {person['confidence']:.2f})")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

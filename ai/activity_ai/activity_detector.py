from activity_ai.person_detector import PersonDetector
from activity_ai.pose_classifier import PoseClassifier

class ActivityDetector:
    """
    활동 중인 사람 감지기
    
    용도: 가정용 공기청정기의 동작 모드 결정
    - 활동 중 (깨어있음) → 공기청정 ON
    - 비활동 (자고있음/없음) → 저전력 모드
    """
    
    def __init__(self, verbose=True):
        """
        Args:
            verbose: 로그 출력 여부
        """
        self.verbose = verbose
        self.detector = PersonDetector(verbose=verbose)
        self.classifier = PoseClassifier(verbose=verbose)
        
        if self.verbose:
            print("[INFO] ActivityDetector initialized")
    
    def is_active_person(self, person):
        """
        한 사람이 활동 중인지 판단
        
        판단 기준:
        - 누워있으면 → 비활동 (자고 있음)
        - 누워있지 않으면 → 활동 중 (서있거나 앉아있음)
        
        Args:
            person: detect_persons() 결과의 한 항목
            
        Returns:
            bool: 활동 중이면 True
        """
        # TODO: 현재는 "누워있지 않으면 활동 중"으로 간단하게 판단
        # TODO: 나중에 필요하면 더 세밀한 기준 추가:
        #   - 움직임 감지 (연속 프레임 비교)
        #   - 특정 자세는 비활동으로 간주 (예: 소파에 기대어 자는 자세)
        #   - 시간 기반 판단 (5분 이상 같은 자세 = 비활동)
        
        # 자세 판별
        pose = self.classifier.classify_pose(person)
        
        # 누워있으면 비활동
        if pose == "lying":
            if self.verbose:
                print(f"  [결과] 비활동 (누워있음)")
            return False
        
        # 서있거나 앉아있으면 활동 중
        # TODO: 의자에 앉음 vs 바닥에 앉음 구분이 필요하면 여기서 처리
        # TODO: Low angle 카메라에서 앉음 판별 정확도 개선 필요
        if self.verbose:
            print(f"  [결과] 활동 중 ({pose})")
        return True
    
    def detect_active_persons(self, image_source):
        """
        이미지에서 활동 중인 사람 찾기
        
        Args:
            image_source: 이미지 파일 경로 또는 numpy array
            
        Returns:
            dict: {
                'total_persons': int,      # 전체 검출된 사람 수
                'active_persons': int,     # 활동 중인 사람 수
                'has_active': bool,        # 활동 중인 사람 있는지
                'persons': list            # 각 사람의 상태 정보
            }
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("활동 중인 사람 감지 시작")
            print('='*60)
        
        # 1. 사람 검출
        persons = self.detector.detect_persons(image_source)
        
        # 2. 신뢰도 필터링
        filtered = self.detector.filter_by_confidence(persons, 0.5, 0.5)
        
        # 3. 각 사람의 활동 상태 판단
        active_count = 0
        person_states = []
        
        for idx, person in enumerate(filtered):
            if self.verbose:
                print(f"\n[Person {idx + 1}]")
            
            is_active = self.is_active_person(person)
            
            person_states.append({
                'person_id': idx + 1,
                'is_active': is_active,
                'confidence': person['confidence'],
                'keypoint_confidence': person['avg_keypoint_conf']
            })
            
            if is_active:
                active_count += 1
        
        # 결과 정리
        result = {
            'total_persons': len(filtered),
            'active_persons': active_count,
            'has_active': active_count > 0,
            'persons': person_states
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"전체 사람: {result['total_persons']}명")
            print(f"활동 중: {result['active_persons']}명")
            print(f"공기청정기 모드: {'공기청정 ON' if result['has_active'] else '저전력 모드'}")
            print('='*60)
        
        return result


# ==================== 테스트 코드 ====================
if __name__ == '__main__':
    print("="*60)
    print("ActivityDetector Test")
    print("="*60)
    
    # 활동 감지기 생성
    detector = ActivityDetector(verbose=True)
    
    # 테스트 케이스
    test_cases = [
        ('test_images/bus.jpg', True, '서있는 사람들'),
        ('test_images/sitting.jpg', True, '앉아있는 사람'),
        ('test_images/lying.jpg', False, '누워있는 사람'),
    ]
    
    print("\n" + "="*60)
    print("테스트 시작")
    print("="*60)
    
    for image_path, expected_active, description in test_cases:
        print(f"\n{'#'*60}")
        print(f"테스트: {description} ({image_path})")
        print('#'*60)
        
        result = detector.detect_active_persons(image_path)
        
        # 결과 검증
        print(f"\n[검증]")
        print(f"예상: {'활동 중' if expected_active else '비활동'}")
        print(f"실제: {'활동 중' if result['has_active'] else '비활동'}")
        
        if result['has_active'] == expected_active:
            print("✅ 정확!")
        else:
            print("❌ 틀림!")
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)

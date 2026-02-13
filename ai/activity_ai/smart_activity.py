from activity_ai.activity_detector import ActivityDetector
from activity_ai.result_filter import ResultFilter
import cv2

class SmartActivityDetector:
    """
    트리거별 활동 감지기
    
    - 일반 순찰: 도착 후 360도만
    - 요리 이벤트: 경로 + 주방 (2개 영상)
    """
    
    def __init__(self, verbose=True):
        """
        Args:
            verbose: 로그 출력 여부
        """
        self.verbose = verbose
        self.detector = ActivityDetector(verbose=False)  # 내부 로그 끄기
        
        if self.verbose:
            print("[INFO] SmartActivityDetector initialized")
    
    def detect_activity_normal(self, video_path):
        """
        일반 순찰 모드
        
        - 360도 회전 영상만 분석
        - 단순 필터링
        
        Args:
            video_path: 360도 영상 파일 경로
            
        Returns:
            dict: {
                'mode': str,  # "공기청정 ON" or "저전력 모드"
                'has_active': bool,
                'confidence': float
            }
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("[NORMAL MODE] 일반 순찰 - 360도 영상 분석")
            print(f"{'='*60}")
            print(f"영상: {video_path}")
        
        # 필터 생성
        simple_filter = ResultFilter(window_size=5, threshold=0.6, verbose=self.verbose)
        
        # 영상 분석
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 분석
            result = self.detector.detect_active_persons(frame)
            simple_filter.add_result(result['has_active'])
            frame_count += 1
        
        cap.release()
        
        if self.verbose:
            print(f"총 {frame_count} 프레임 분석 완료")
        
        # 최종 판단
        final = simple_filter.get_filtered_result()
        
        mode = "공기청정 ON" if final['has_active'] else "저전력 모드"
        
        if self.verbose:
            print(f"\n최종 결과: {mode}")
            print(f"신뢰도: {final['confidence']*100:.1f}%")
            print('='*60)
        
        return {
            'mode': mode,
            'has_active': final['has_active'],
            'confidence': final['confidence']
        }
    
    def detect_cooking_event(self, corridor_video, kitchen_video):
        """
        요리 이벤트 모드
        
        - 경로 영상 + 주방 영상 분석
        - 구간별 필터링
        - OR 조건 판단
        
        Args:
            corridor_video: 복도 이동 영상 경로
            kitchen_video: 주방 360도 영상 경로
            
        Returns:
            dict: {
                'confirmed': bool,  # 요리 이벤트 확정 여부
                'corridor_confidence': float,
                'kitchen_confidence': float,
                'reason': str  # 판단 이유
            }
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("[COOKING MODE] 요리 이벤트 감지")
            print(f"{'='*60}")
        
        # 1. 복도 영상 분석
        if self.verbose:
            print(f"\n[1/2] 복도 영상 분석: {corridor_video}")
        
        corridor_filter = ResultFilter(window_size=10, threshold=0.6, verbose=self.verbose)
        
        cap = cv2.VideoCapture(corridor_video)
        corridor_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.detector.detect_active_persons(frame)
            corridor_filter.add_result(result['has_active'])
            corridor_frames += 1
        
        cap.release()
        
        corridor_result = corridor_filter.get_filtered_result()
        
        if self.verbose:
            print(f"복도: {corridor_frames} 프레임 분석")
            print(f"복도 신뢰도: {corridor_result['confidence']*100:.1f}%")
        
        # 2. 주방 영상 분석
        if self.verbose:
            print(f"\n[2/2] 주방 영상 분석: {kitchen_video}")
        
        kitchen_filter = ResultFilter(window_size=5, threshold=0.6, verbose=self.verbose)
        
        cap = cv2.VideoCapture(kitchen_video)
        kitchen_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.detector.detect_active_persons(frame)
            kitchen_filter.add_result(result['has_active'])
            kitchen_frames += 1
        
        cap.release()
        
        kitchen_result = kitchen_filter.get_filtered_result()
        
        if self.verbose:
            print(f"주방: {kitchen_frames} 프레임 분석")
            print(f"주방 신뢰도: {kitchen_result['confidence']*100:.1f}%")
        
        # 3. OR 조건 판단
        corridor_active = corridor_result['has_active']
        kitchen_active = kitchen_result['has_active']
        
        confirmed = corridor_active or kitchen_active
        
        # 판단 이유
        if corridor_active and kitchen_active:
            reason = "복도와 주방 모두에서 활동 중인 사람 감지"
        elif corridor_active:
            reason = "복도에서 활동 중인 사람 감지 (주방은 미감지)"
        elif kitchen_active:
            reason = "주방에서 활동 중인 사람 감지 (복도는 미감지)"
        else:
            reason = "활동 중인 사람 미감지"
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("최종 판단:")
            print(f"  복도: {'활동 중' if corridor_active else '비활동'} ({corridor_result['confidence']*100:.1f}%)")
            print(f"  주방: {'활동 중' if kitchen_active else '비활동'} ({kitchen_result['confidence']*100:.1f}%)")
            print(f"  결과: {'요리 이벤트 확정 ✅' if confirmed else '요리 이벤트 아님 ❌'}")
            print(f"  이유: {reason}")
            print('='*60)
        
        return {
            'confirmed': confirmed,
            'corridor_confidence': corridor_result['confidence'],
            'kitchen_confidence': kitchen_result['confidence'],
            'reason': reason
        }


# ==================== 테스트 코드 ====================
if __name__ == '__main__':
    print("="*60)
    print("SmartActivityDetector Test")
    print("="*60)
    
    detector = SmartActivityDetector(verbose=True)
    
    # 시나리오 1: 일반 순찰 (bus.jpg를 영상처럼 사용)
    print("\n[Test 1] 일반 순찰 모드")
    print("-"*60)
    print("NOTE: 실제로는 360도 영상을 사용")
    print("      테스트에서는 이미지로 대체")
    print()
    
    # 이미지를 영상처럼 처리 (임시)
    result = detector.detector.detect_active_persons('test_images/bus.jpg')
    print(f"결과: {'공기청정 ON' if result['has_active'] else '저전력 모드'}")
    
    # 시나리오 2: 요리 이벤트 (개념 설명)
    print(f"\n{'='*60}")
    print("[Test 2] 요리 이벤트 모드")
    print("-"*60)
    print("NOTE: 실제 영상 파일이 필요합니다")
    print()
    print("사용법:")
    print("  result = detector.detect_cooking_event(")
    print("      corridor_video='corridor.mp4',")
    print("      kitchen_video='kitchen.mp4'")
    print("  )")
    print()
    print("  if result['confirmed']:")
    print("      print('요리 이벤트 확정!')")
    print("      # 영상 저장")
    print("  else:")
    print("      print('요리 이벤트 아님')")
    print("      # 영상 삭제")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

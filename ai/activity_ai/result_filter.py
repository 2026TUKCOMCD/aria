import numpy as np
from collections import deque

class ResultFilter:
    """
    활동 판단 결과 필터링
    
    여러 프레임의 결과를 모아서 안정적인 판단
    """
    
    def __init__(self, window_size=5, threshold=0.6, verbose=True):
        """
        Args:
            window_size: 필터링할 프레임 수 (기본 5)
            threshold: 활동 중 판단 기준 (기본 60%, 즉 5장 중 3장)
            verbose: 로그 출력 여부
        """
        self.window_size = window_size
        self.threshold = threshold
        self.verbose = verbose
        
        # 최근 결과 저장 (FIFO)
        self.history = deque(maxlen=window_size)
        
        if self.verbose:
            print(f"[INFO] ResultFilter initialized")
            print(f"  - window_size: {window_size} frames")
            print(f"  - threshold: {threshold*100:.0f}%")
    
    def add_result(self, has_active):
        """
        새로운 결과 추가
        
        Args:
            has_active: bool, 활동 중인 사람 있는지
            
        Returns:
            None
        """
        self.history.append(has_active)
        
        if self.verbose:
            print(f"[FILTER] Added result: {has_active}, "
                  f"history size: {len(self.history)}/{self.window_size}")
    
    def get_filtered_result(self):
        """
        필터링된 최종 결과 반환
        
        Returns:
            dict: {
                'has_active': bool,  # 필터링된 최종 판단
                'confidence': float,  # 신뢰도 (0.0~1.0)
                'history': list,     # 최근 결과 히스토리
                'sample_count': int  # 현재 샘플 수
            }
        """
        if len(self.history) == 0:
            return {
                'has_active': False,
                'confidence': 0.0,
                'history': [],
                'sample_count': 0
            }
        
        # 활동 중 비율 계산
        active_count = sum(self.history)
        total_count = len(self.history)
        confidence = active_count / total_count
        
        # 임계값 기준 판단
        has_active = confidence >= self.threshold
        
        if self.verbose:
            print(f"[FILTER] Filtered result:")
            print(f"  - Active: {active_count}/{total_count} frames")
            print(f"  - Confidence: {confidence*100:.1f}%")
            print(f"  - Decision: {'ACTIVE' if has_active else 'INACTIVE'}")
        
        return {
            'has_active': has_active,
            'confidence': confidence,
            'history': list(self.history),
            'sample_count': total_count
        }
    
    def reset(self):
        """히스토리 초기화"""
        self.history.clear()
        if self.verbose:
            print("[FILTER] History reset")


# ==================== 테스트 코드 ====================
if __name__ == '__main__':
    print("="*60)
    print("ResultFilter Test")
    print("="*60)
    
    # 필터 생성
    filter = ResultFilter(window_size=5, threshold=0.6, verbose=True)
    
    # 시나리오 1: 안정적인 활동 중
    print("\n[Scenario 1] 안정적인 활동 중")
    print("-"*60)
    test_results_1 = [True, True, True, True, True]
    for result in test_results_1:
        filter.add_result(result)
    
    final = filter.get_filtered_result()
    print(f"\n최종 판단: {'활동 중' if final['has_active'] else '비활동'}")
    print(f"신뢰도: {final['confidence']*100:.1f}%\n")
    
    # 시나리오 2: 순간 오류 (5장 중 1장만 실패)
    filter.reset()
    print("\n[Scenario 2] 순간 오류 (5장 중 1장 실패)")
    print("-"*60)
    test_results_2 = [True, True, False, True, True]  # 1장 오류
    for result in test_results_2:
        filter.add_result(result)
    
    final = filter.get_filtered_result()
    print(f"\n최종 판단: {'활동 중' if final['has_active'] else '비활동'}")
    print(f"신뢰도: {final['confidence']*100:.1f}%")
    print(f"→ 오류 무시하고 활동 중으로 판단!\n")
    
    # 시나리오 3: 애매한 경우 (5장 중 3장 활동)
    filter.reset()
    print("\n[Scenario 3] 애매한 경우 (5장 중 3장 활동)")
    print("-"*60)
    test_results_3 = [True, True, True, False, False]
    for result in test_results_3:
        filter.add_result(result)
    
    final = filter.get_filtered_result()
    print(f"\n최종 판단: {'활동 중' if final['has_active'] else '비활동'}")
    print(f"신뢰도: {final['confidence']*100:.1f}%")
    print(f"→ 60% 임계값으로 활동 중 판단!\n")
    
    # 시나리오 4: 비활동
    filter.reset()
    print("\n[Scenario 4] 비활동")
    print("-"*60)
    test_results_4 = [False, False, False, False, False]
    for result in test_results_4:
        filter.add_result(result)
    
    final = filter.get_filtered_result()
    print(f"\n최종 판단: {'활동 중' if final['has_active'] else '비활동'}")
    print(f"신뢰도: {final['confidence']*100:.1f}%\n")
    
    print("="*60)
    print("Test completed!")
    print("="*60)

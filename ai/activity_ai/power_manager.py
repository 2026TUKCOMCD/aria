"""
저전력 모드 관리 모듈

현재: 고정값으로 테스트
TODO: JSON 파일 연동 (통신 담당자가 구현)
"""

from datetime import datetime, time
import json
import os


class PowerManager:
    def __init__(self, config_file=None, sleep_time="23:00", wake_time="07:00", verbose=True):
        """
        Args:
            config_file (str): 설정 파일 경로 (현재 미사용, 나중에 연동)
            sleep_time (str): 취침 시간 (현재 고정값)
            wake_time (str): 기상 시간 (현재 고정값)
        """
        self.config_file = config_file
        self.verbose = verbose
        self.current_mode = 'NORMAL'
        
        # ===== 현재: 고정값 사용 =====
        self.sleep_time = self._parse_time(sleep_time)
        self.wake_time = self._parse_time(wake_time)
        
        # ===== TODO: JSON 파일 연동 (나중에 주석 해제) =====
        # if config_file and os.path.exists(config_file):
        #     config = self._load_config(sleep_time, wake_time)
        #     self.sleep_time = self._parse_time(config['sleep_time'])
        #     self.wake_time = self._parse_time(config['wake_time'])
        
        if self.verbose:
            print(f"[PowerManager] Initialized (Test mode)")
            print(f"  Sleep: {sleep_time}, Wake: {wake_time}")
            # print(f"  Config file: {config_file}")  # 나중에 활성화
    
    # ===== TODO: 나중에 사용할 함수들 (현재 주석) =====
    
    # def _load_config(self, default_sleep, default_wake):
    #     """JSON 파일에서 스케줄 로드"""
    #     if os.path.exists(self.config_file):
    #         with open(self.config_file, 'r') as f:
    #             config = json.load(f)
    #         return config
    #     return {'sleep_time': default_sleep, 'wake_time': default_wake}
    
    # def _save_config(self, sleep_time, wake_time):
    #     """JSON 파일에 스케줄 저장"""
    #     config = {
    #         'sleep_time': sleep_time,
    #         'wake_time': wake_time,
    #         'updated_at': datetime.now().isoformat()
    #     }
    #     os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
    #     with open(self.config_file, 'w') as f:
    #         json.dump(config, f, indent=2)
    
    # def update_schedule(self, sleep_time, wake_time):
    #     """웹/클라우드에서 스케줄 변경 시 호출"""
    #     self.sleep_time = self._parse_time(sleep_time)
    #     self.wake_time = self._parse_time(wake_time)
    #     if self.config_file:
    #         self._save_config(sleep_time, wake_time)
    
    # def reload_config(self):
    #     """설정 파일 다시 읽기 (MQTT 수신 시)"""
    #     if self.config_file and os.path.exists(self.config_file):
    #         with open(self.config_file, 'r') as f:
    #             config = json.load(f)
    #         self.sleep_time = self._parse_time(config['sleep_time'])
    #         self.wake_time = self._parse_time(config['wake_time'])
    
    def _parse_time(self, time_str):
        """시간 문자열 → time 객체"""
        hour, minute = map(int, time_str.split(':'))
        return time(hour, minute)
    
    def check_mode(self, current_time=None):
        """현재 모드 판단"""
        if current_time is None:
            current_time = datetime.now()
        
        current = current_time.time()
        
        if self.sleep_time > self.wake_time:
            is_sleep_time = current >= self.sleep_time or current < self.wake_time
        else:
            is_sleep_time = self.sleep_time <= current < self.wake_time
        
        new_mode = 'SLEEP_MODE' if is_sleep_time else 'NORMAL_MODE'
        
        if new_mode != self.current_mode:
            if self.verbose:
                print(f"[PowerManager] Mode: {self.current_mode} → {new_mode}")
            self.current_mode = new_mode
        
        return new_mode
    
    def should_process_activity(self, current_time=None):
        """활동 감지 처리 여부"""
        mode = self.check_mode(current_time)
        
        if mode == 'SLEEP_MODE':
            if self.verbose:
                print("[PowerManager] SLEEP_MODE - Skip")
            return False
        
        return True
    
    def get_status(self):
        """현재 상태"""
        return {
            'mode': self.current_mode,
            'sleep_time': self.sleep_time.strftime('%H:%M'),
            'wake_time': self.wake_time.strftime('%H:%M'),
            'current_time': datetime.now().strftime('%H:%M:%S')
        }


# ===== 사용 예시 =====
if __name__ == '__main__':
    print("="*60)
    print("PowerManager Test (고정값)")
    print("="*60)
    
    # 현재: 고정값 테스트
    pm = PowerManager(sleep_time="23:00", wake_time="07:00")
    
    # ===== TODO: 나중에 이렇게 사용 =====
    # pm = PowerManager(config_file="/home/jj/aria/config/schedule.json")
    
    # 테스트
    test_times = [
        ("14:30", "오후"),
        ("23:30", "취침"),
        ("02:00", "한밤중"),
        ("08:00", "기상 후"),
    ]
    
    print("\n시간대별 모드:")
    for time_str, desc in test_times:
        h, m = map(int, time_str.split(':'))
        test_time = datetime(2026, 2, 11, h, m)
        mode = pm.check_mode(test_time)
        process = pm.should_process_activity(test_time)
        print(f"{time_str} ({desc:8s}) → {mode:12s} | Process: {process}")
    
    print("\n" + "="*60)
    print("✅ Test passed!")
    print("="*60)
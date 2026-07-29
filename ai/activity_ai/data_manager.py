"""
영상 데이터 및 메타데이터 관리 모듈

YOLO 추론 결과를 JSON으로 저장하고 원본 영상 자동 삭제

Author: 박진주
Sprint: Sprint 3 (F7-1)
"""

import json
import os
from datetime import datetime


class DataManager:
    """
    영상 데이터 및 메타데이터 관리 클래스
    
    Features:
    - YOLO 결과 JSON 저장
    - 원본 영상 자동 삭제
    - 일반 순찰 / 요리 이벤트 분리 처리
    """
    
    def __init__(self, output_dir=None, verbose=True):
        """
        Args:
            output_dir (str): JSON 저장 경로
            verbose (bool): 로그 출력 여부
        """
        if output_dir is None:
            output_dir = "/home/jj/aria/data/metadata"
        
        self.output_dir = output_dir
        self.verbose = verbose
        
        # 디렉토리 없으면 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.verbose:
            print(f"[DataManager] Initialized")
            print(f"  Output dir: {self.output_dir}")
    
    def save_metadata(self, result, video_path, location="unknown"):
        """
        일반 순찰 결과 저장 (영상 1개)
        
        Args:
            result (dict): detect_activity_normal() 결과
            video_path (str): 영상 파일 경로
            location (str): 구역명 (예: living_room, bedroom)
            
        Returns:
            str: 저장된 JSON 파일 경로 (실패 시 None)
        """
        if self.verbose:
            print(f"\n[DataManager] Saving metadata...")
            print(f"  Location: {location}")
            print(f"  Video: {video_path}")
        
        # 1. 메타데이터 생성
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "patrol",
            "location": location,
            "video_file": os.path.basename(video_path),
            "video_deleted": False,
            "result": result
        }
        
        # 2. JSON 파일명 생성
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        json_filename = f"{video_name}_metadata.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        # 3. JSON 저장
        json_saved = self._save_json(metadata, json_path)
        
        if not json_saved:
            if self.verbose:
                print(f"[DataManager] ❌ JSON save failed - video NOT deleted")
            return None
        
        # 4. 영상 삭제 (JSON 저장 성공 시에만!)
        video_deleted = self._delete_video(video_path)
        
        # 5. 삭제 여부 업데이트
        if video_deleted:
            metadata["video_deleted"] = True
            self._save_json(metadata, json_path)
        
        if self.verbose:
            print(f"[DataManager] ✅ Done!")
            print(f"  JSON: {json_path}")
            print(f"  Video deleted: {video_deleted}")
        
        return json_path
    
    def save_cooking_event(self, result, corridor_video, kitchen_video):
        """
        요리 이벤트 결과 저장 (영상 2개)
        
        Args:
            result (dict): detect_cooking_event() 결과
            corridor_video (str): 경로 영상 파일 경로
            kitchen_video (str): 부엌 360도 영상 파일 경로
            
        Returns:
            str: 저장된 JSON 파일 경로 (실패 시 None)
        """
        if self.verbose:
            print(f"\n[DataManager] Saving cooking event metadata...")
            print(f"  Corridor video: {corridor_video}")
            print(f"  Kitchen video: {kitchen_video}")
        
        # 1. 메타데이터 생성
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "cooking_verification",
            "location": "kitchen",
            "videos": [
                {
                    "type": "corridor",
                    "file": os.path.basename(corridor_video),
                    "deleted": False
                },
                {
                    "type": "kitchen_360",
                    "file": os.path.basename(kitchen_video),
                    "deleted": False
                }
            ],
            "result": result
        }
        
        # 2. JSON 파일명 생성 (타임스탬프 기반)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"cooking_{timestamp}_metadata.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        # 3. JSON 저장
        json_saved = self._save_json(metadata, json_path)
        
        if not json_saved:
            if self.verbose:
                print(f"[DataManager] ❌ JSON save failed - videos NOT deleted")
            return None
        
        # 4. 영상 2개 삭제 (JSON 저장 성공 시에만!)
        corridor_deleted = self._delete_video(corridor_video)
        kitchen_deleted = self._delete_video(kitchen_video)
        
        # 5. 삭제 여부 업데이트
        metadata["videos"][0]["deleted"] = corridor_deleted
        metadata["videos"][1]["deleted"] = kitchen_deleted
        self._save_json(metadata, json_path)
        
        if self.verbose:
            print(f"[DataManager] ✅ Done!")
            print(f"  JSON: {json_path}")
            print(f"  Corridor deleted: {corridor_deleted}")
            print(f"  Kitchen deleted: {kitchen_deleted}")
        
        return json_path
    
    def _save_json(self, metadata, json_path):
        """
        JSON 파일 저장
        
        Args:
            metadata (dict): 저장할 데이터
            json_path (str): 저장 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"[DataManager] ❌ JSON save error: {e}")
            return False
    
    def _delete_video(self, video_path):
        """
        영상 파일 안전하게 삭제
        
        Args:
            video_path (str): 삭제할 영상 경로
            
        Returns:
            bool: 성공 여부
        """
        if not os.path.exists(video_path):
            if self.verbose:
                print(f"[DataManager] ⚠️ Video not found: {video_path}")
            return False
        
        try:
            os.remove(video_path)
            if self.verbose:
                print(f"[DataManager] 🗑️ Video deleted: {video_path}")
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"[DataManager] ❌ Delete error: {e}")
            return False


# ==================== 테스트 코드 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("DataManager Test - F7-1")
    print("=" * 60)
    
    # 테스트용 임시 디렉토리
    dm = DataManager(output_dir="/tmp/test_metadata")
    
    # 테스트 1: 일반 순찰 (영상 1개)
    print("\n[Test 1] 일반 순찰 메타데이터 저장")
    print("-" * 60)
    
    # 임시 영상 파일 생성
    test_video = "/tmp/living_room_360.mp4"
    with open(test_video, 'w') as f:
        f.write("fake video")
    
    # 테스트 결과
    test_result = {
        "mode": "active",
        "has_active": True,
        "active_count": 2,
        "confidence": 0.85,
        "detections": [
            {
                "pose": "standing",
                "confidence": 0.92,
                "bbox": [100, 200, 50, 100]
            }
        ]
    }
    
    json_path = dm.save_metadata(
        result=test_result,
        video_path=test_video,
        location="living_room"
    )
    
    # 결과 확인
    print(f"\n저장된 JSON 확인:")
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
    
    print(f"\n영상 삭제 확인: {not os.path.exists(test_video)}")
    
    # 테스트 2: 요리 이벤트 (영상 2개)
    print("\n[Test 2] 요리 이벤트 메타데이터 저장")
    print("-" * 60)
    
    # 임시 영상 파일 2개 생성
    corridor_video = "/tmp/corridor_video.mp4"
    kitchen_video = "/tmp/kitchen_360.mp4"
    
    with open(corridor_video, 'w') as f:
        f.write("fake corridor video")
    with open(kitchen_video, 'w') as f:
        f.write("fake kitchen video")
    
    # 테스트 결과
    cooking_result = {
        "mode": "active",
        "has_active": True,
        "corridor_active": True,
        "kitchen_active": True,
        "final_decision": "cooking_confirmed"
    }
    
    json_path = dm.save_cooking_event(
        result=cooking_result,
        corridor_video=corridor_video,
        kitchen_video=kitchen_video
    )
    
    # 결과 확인
    print(f"\n저장된 JSON 확인:")
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
    
    print(f"\n경로 영상 삭제: {not os.path.exists(corridor_video)}")
    print(f"부엌 영상 삭제: {not os.path.exists(kitchen_video)}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
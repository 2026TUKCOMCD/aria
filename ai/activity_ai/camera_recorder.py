"""
Picamera2 기반 카메라 녹화 모듈

라즈베리파이 카메라 v2 + Picamera2

Author: 박진주
Sprint: Sprint 3
"""

import cv2
import os
import time
import threading
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput


class CameraRecorder:
    """
    Picamera2 기반 카메라 녹화
    
    Features:
    - 360도 회전 녹화 (일반 순찰)
    - 경로 녹화 (요리 이벤트)
    """
    
    def __init__(self, output_dir=None, resolution=(1640, 1232), fps=15, verbose=True):
        """
        Args:
            output_dir (str): 영상 저장 경로
            resolution (tuple): 해상도 (width, height)
            fps (int): 프레임 레이트
            verbose (bool): 로그 출력
        """
        if output_dir is None:
            output_dir = "/home/jj/aria/data/videos"
        
        self.output_dir = output_dir
        self.resolution = resolution
        self.fps = fps
        self.verbose = verbose
        
        # 녹화 상태
        self.is_recording = False
        self.current_video_path = None
        self.recording_start_time = None
        
        # 카메라 객체
        self.camera = None
        self.writer = None
        self.record_thread = None
        
        # 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.verbose:
            print(f"[CameraRecorder] Picamera2 모드")
            print(f"  Output: {self.output_dir}")
            print(f"  Resolution: {self.resolution}")
            print(f"  FPS: {self.fps}")
    
    def initialize_camera(self):
        """
        카메라 초기화
        
        Returns:
            bool: 성공 여부
        """
        try:
            if self.camera is not None:
                if self.verbose:
                    print("[CameraRecorder] Camera already opened")
                return True
            
            self.camera = Picamera2()
            
            # 카메라 설정
            config = self.camera.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={"FrameRate": self.fps}
            )
            self.camera.configure(config)
            self.camera.start()
            
            # 최대 화각으로 설정 (줌아웃 - ScalerCrop)
            try:
                size = self.camera.camera_properties['PixelArraySize']
                self.camera.set_controls({"ScalerCrop": (0, 0, size[0], size[1])})
                if self.verbose:
                    print(f"[CameraRecorder] 📐 ScalerCrop 적용 (최대 화각: {size[0]}x{size[1]})")
            except Exception as e:
                if self.verbose:
                    print(f"[CameraRecorder] ⚠️ ScalerCrop 실패 (무시): {e}")
            
            # 워밍업
            time.sleep(0.5)
            
            # 테스트 프레임
            frame = self.camera.capture_array()
            if frame is None:
                raise Exception("프레임 읽기 실패")
            
            if self.verbose:
                print(f"[CameraRecorder] ✅ Camera OK ({self.resolution[0]}x{self.resolution[1]})")
            
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"[CameraRecorder] ❌ Camera init failed: {e}")
            return False
    
    def start_recording(self, filename=None, mode="360"):
        """
        녹화 시작
        
        Args:
            filename (str): 파일명 (None이면 자동)
            mode (str): "360" (360도) 또는 "corridor" (경로)
            
        Returns:
            str: 녹화 파일 경로 (실패 시 None)
        """
        if self.is_recording:
            if self.verbose:
                print("[CameraRecorder] ⚠️ Already recording")
            return None
        
        # 카메라 초기화
        if not self.initialize_camera():
            return None
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{mode}_{timestamp}.mp4"
        
        self.current_video_path = os.path.join(self.output_dir, filename)
        
        try:
            # VideoWriter 설정 (OpenCV로 mp4 저장)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.current_video_path,
                fourcc,
                self.fps,
                self.resolution
            )
            
            if not self.writer.isOpened():
                raise Exception("VideoWriter 초기화 실패")
            
            self.is_recording = True
            self.recording_start_time = time.time()
            
            # 녹화 스레드 시작
            self.record_thread = threading.Thread(target=self._record_loop)
            self.record_thread.daemon = True
            self.record_thread.start()
            
            if self.verbose:
                print(f"[CameraRecorder] 🔴 Recording started")
                print(f"  Mode: {mode}")
                print(f"  File: {self.current_video_path}")
            
            return self.current_video_path
        
        except Exception as e:
            if self.verbose:
                print(f"[CameraRecorder] ❌ Recording start failed: {e}")
            return None
    
    def _record_loop(self):
        """녹화 루프 (별도 스레드)"""
        frame_interval = 1.0 / self.fps
        
        while self.is_recording:
            # picamera2로 프레임 캡처
            frame = self.camera.capture_array()
            
            if frame is not None:
                # RGB -> BGR 변환 (OpenCV VideoWriter용)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.writer.write(frame_bgr)
            else:
                if self.verbose:
                    print("[CameraRecorder] ⚠️ 프레임 읽기 실패")
                break
            
            time.sleep(frame_interval)
    
    def stop_recording(self):
        """
        녹화 정지
        
        Returns:
            str: 녹화된 파일 경로 (실패 시 None)
        """
        if not self.is_recording:
            if self.verbose:
                print("[CameraRecorder] ⚠️ Not recording")
            return None
        
        # 녹화 중단
        self.is_recording = False
        
        # 스레드 종료 대기
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        
        # VideoWriter 해제
        if self.writer:
            self.writer.release()
            self.writer = None
        
        duration = time.time() - self.recording_start_time
        video_path = self.current_video_path
        self.current_video_path = None
        
        if self.verbose:
            print(f"[CameraRecorder] ⏹️ Recording stopped")
            print(f"  Duration: {duration:.1f}s")
            print(f"  File: {video_path}")
        
        return video_path
    
    def cleanup(self):
        """리소스 정리"""
        if self.is_recording:
            self.stop_recording()
        
        if self.camera:
            self.camera.stop()
            self.camera = None
        
        if self.verbose:
            print("[CameraRecorder] Cleanup done")
    
    def __del__(self):
        """소멸자"""
        self.cleanup()


# ==================== 통합 예시 ====================
def integration_example():
    """사용 예시"""
    
    from smart_activity import detect_activity_normal, detect_cooking_event
    from data_manager import DataManager
    
    recorder = CameraRecorder()
    dm = DataManager()
    
    # 시나리오 1: 일반 순찰 (360도 1개)
    print("\n[Scenario 1] 일반 순찰 - 360도 녹화")
    
    video = recorder.start_recording(mode="360")
    time.sleep(10)  # 360도 회전 시간
    video = recorder.stop_recording()
    
    result = detect_activity_normal(video)
    dm.save_metadata(result, video, location="living_room")
    
    # 시나리오 2: 요리 이벤트 (경로 + 360도 = 2개)
    print("\n[Scenario 2] 요리 이벤트 - 2개 영상")
    
    # 경로 녹화
    corridor = recorder.start_recording(mode="corridor")
    time.sleep(3)
    corridor = recorder.stop_recording()
    
    # 360도 녹화
    kitchen = recorder.start_recording(mode="360")
    time.sleep(10)
    kitchen = recorder.stop_recording()
    
    result = detect_cooking_event(corridor, kitchen)
    dm.save_cooking_event(result, corridor, kitchen)
    
    recorder.cleanup()


# ==================== 테스트 ====================
if __name__ == '__main__':
    print("="*60)
    print("Picamera2 Camera Test")
    print("="*60)
    
    try:
        recorder = CameraRecorder(output_dir="/tmp/test_videos")
        
        # 3초 녹화 테스트
        print("\n3초 녹화 시작...")
        video = recorder.start_recording(mode="test")
        
        if video:
            time.sleep(3)
            video = recorder.stop_recording()
            print(f"\n저장됨: {video}")
        else:
            print("\n❌ 녹화 실패")
        
        recorder.cleanup()
        
        print("\n" + "="*60)
        print("Test completed!")
        print("="*60)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
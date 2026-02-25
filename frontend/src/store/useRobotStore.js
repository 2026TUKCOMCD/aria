import { create } from 'zustand';

// 1. 상태들의 타입을 정의합니다. (인터페이스)
interface RobotState {
  isAiMode: boolean;
  robotStatus: 'IDLE' | 'MOVING' | 'COOKING';
  battery: number;
  airQuality: {
    pm25: number;
    voc: number;
  };
  // 2. 액션(함수)들의 타입 정의
  setAiMode: (mode: boolean) => void;
  setAirQuality: (data: { pm25: number; voc: number }) => void;
  updateBattery: (val: number) => void;
}

// 3. 정의한 타입을 적용하여 스토어 생성
const useRobotStore = create<RobotState>((set) => ({
  isAiMode: true,
  robotStatus: 'IDLE',
  battery: 100,
  airQuality: { pm25: 0, voc: 0 },

  setAiMode: (mode) => set({ isAiMode: mode }),
  setAirQuality: (data) => set({ airQuality: data }),
  updateBattery: (val) => set({ battery: val }),
}));

export default useRobotStore;
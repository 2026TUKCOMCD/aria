import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// 1. 로그 데이터 타입 정의
interface Log {
  time: string;
  content: string;
}

// 2. 스토어 상태 및 액션 타입 정의
interface RobotState {
  isAiMode: boolean;
  isRunning: boolean;
  robotStatus: 'IDLE' | 'MOVING' | 'COOKING';
  battery: number;
  airQuality: {
    pm25: number;
    voc: number;
  };
  logs: Log[]; // 로그 배열

  // 액션들
  setAiMode: (mode: boolean) => void;
  setIsRunning: (status: boolean) => void;
  setAirQuality: (data: { pm25: number; voc: number }) => void;
  updateBattery: (val: number) => void;
  
  // 로그 관련 액션
  addLog: (message: string) => void;        // 실시간 로그 추가용
  clearLogs: () => void;                   // 로그 초기화용
  fetchLogs: (robotId: string) => Promise<void>; // [핵심] 서버 C파트 호출용
}

// 3. 스토어 생성
const useRobotStore = create<RobotState>()(
  persist(
    (set) => ({
      isAiMode: true,
      isRunning: false,
      robotStatus: 'IDLE',
      battery: 100,
      airQuality: { pm25: 0, voc: 0 },
      logs: [],

      setAiMode: (mode) => set({ isAiMode: mode }),
      setIsRunning: (status) => set({ isRunning: status }),
      setAirQuality: (data) => set({ airQuality: data }),
      updateBattery: (val) => set({ battery: val }),

      // [A] 실시간 알림을 로그 맨 앞에 추가 (Socket.io용)
      addLog: (message) => set((state) => {
        const now = new Date();
        const timestamp = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
        
        const newLog = { time: timestamp, content: message };
        // 최신 로그 20개까지만 유지
        return { logs: [newLog, ...state.logs].slice(0, 20) };
      }),

      // [B] 서버(C파트)에서 과거 로그 가져오기
      fetchLogs: async (robotId) => {
        try {
          console.log(`서버에서 ${robotId}의 과거 로그를 조회합니다...`);
          
          // ngrok 주소나 localhost:3000 사용
          const response = await fetch(`http://localhost:3000/api/events?robot_id=${robotId}`);
          const result = await response.json();

          if (result.success && result.data) {
            // DB 결과(row)를 UI용 포맷 { time, content }으로 변환
            const formattedLogs = result.data.map((row: any) => {
              const date = new Date(row.created_at);
              return {
                time: `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`,
                content: row.message
              };
            });

            set({ logs: formattedLogs });
            console.log("과거 로그 로드 완료!");
          }
        } catch (error) {
          console.error("서버 C파트 호출 실패 (DB 연결 확인 필요):", error);
          // DB 연결이 안될 때 테스트를 위한 기본 문구 유지 (선택 사항)
        }
      },

      clearLogs: () => set({ logs: [] }),
    }),
    {
      name: 'aria-robot-storage', // 로컬스토리지 저장 키
    }
  )
);

export default useRobotStore;
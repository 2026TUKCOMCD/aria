import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  fetchRobotMap,
  fetchRobotStatus,
  fetchRobotZones,
  fetchZoneAirQuality,
  saveRobotZones,
  type RobotMap,
  type RobotStatusSummary,
  type RobotZone,
  type ZoneAirQuality,
} from '../api/ARIARobotController';

interface Log {
  time: string;
  content: string;
}

export interface RobotPosition {
  robot_id: string;
  x: number;
  y: number;
  theta: number;
  updated_at: string;
}

interface RobotState {
  isAiMode: boolean;
  isRunning: boolean;
  robotStatus: 'IDLE' | 'MOVING' | 'COOKING';
  battery: number;
  airQuality: {
    pm25: number;
    voc: number;
  };
  robotStatusSummary: RobotStatusSummary | null;
  robotStatusError: string | null;
  robotPosition: RobotPosition | null;
  logs: Log[];
  mapData: RobotMap | null;
  zones: RobotZone[];
  zoneAirQuality: ZoneAirQuality[];
  airQualityUpdatedAt: string | null;
  airQualityError: string | null;
  isMapLoading: boolean;
  mapError: string | null;

  setAiMode: (mode: boolean) => void;
  setIsRunning: (status: boolean) => void;
  setAirQuality: (data: { pm25: number; voc: number }) => void;
  updateBattery: (val: number) => void;
  setRobotPosition: (position: RobotPosition) => void;
  loadRobotStatus: (robotId?: string) => Promise<void>;
  addLog: (message: string) => void;
  clearLogs: () => void;
  fetchLogs: (robotId: string) => Promise<void>;
  loadMapData: (robotId?: string) => Promise<void>;
  loadZones: (robotId?: string) => Promise<void>;
  loadZoneAirQuality: (robotId?: string) => Promise<void>;
  updateZone: (zone: RobotZone) => void;
  saveZones: (robotId?: string) => Promise<void>;
}

const useRobotStore = create<RobotState>()(
  persist(
    (set, get) => ({
      isAiMode: true,
      isRunning: false,
      robotStatus: 'IDLE',
      battery: 100,
      airQuality: { pm25: 0, voc: 0 },
      robotStatusSummary: null,
      robotStatusError: null,
      robotPosition: null,
      logs: [],
      mapData: null,
      zones: [],
      zoneAirQuality: [],
      airQualityUpdatedAt: null,
      airQualityError: null,
      isMapLoading: false,
      mapError: null,

      setAiMode: (mode) => set({ isAiMode: mode }),
      setIsRunning: (status) => set({ isRunning: status }),
      setAirQuality: (data) => set({ airQuality: data }),
      updateBattery: (val) => set({ battery: val }),
      setRobotPosition: (position) => set({ robotPosition: position }),

      loadRobotStatus: async (robotId) => {
        try {
          const robotStatusSummary = await fetchRobotStatus(robotId);
          set({
            robotStatusSummary,
            robotStatusError: null,
            battery: robotStatusSummary.robot_status.battery,
            airQuality: {
              pm25: robotStatusSummary.air_quality.sensors.pm25,
              voc: robotStatusSummary.air_quality.sensors.voc,
            },
            isRunning: robotStatusSummary.robot_status.power === 'ON',
            isAiMode: robotStatusSummary.robot_status.mode === 'AUTO',
          });
        } catch (error) {
          console.error('로봇 상태 조회 실패:', error);
          set({ robotStatusError: '로봇 상태를 불러오지 못했습니다.' });
        }
      },

      addLog: (message) => set((state) => {
        const now = new Date();
        const timestamp = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
        const newLog = { time: timestamp, content: message };

        return { logs: [newLog, ...state.logs].slice(0, 20) };
      }),

      fetchLogs: async (robotId) => {
        try {
          const response = await fetch(`http://localhost:3000/api/events?robot_id=${robotId}`);
          const result = await response.json();

          if (result.success && result.data) {
            const formattedLogs = result.data.map((row: any) => {
              const date = new Date(row.created_at);
              return {
                time: `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`,
                content: row.message,
              };
            });

            set({ logs: formattedLogs });
          }
        } catch (error) {
          console.error('이벤트 로그 조회 실패:', error);
        }
      },

      loadMapData: async (robotId) => {
        const previousMap = get().mapData;
        set({ isMapLoading: true, mapError: null });

        try {
          const mapData = await fetchRobotMap(robotId);
          set({ mapData, isMapLoading: false, mapError: null });
        } catch (error) {
          console.error('맵 데이터 조회 실패:', error);
          set({
            mapData: previousMap,
            isMapLoading: false,
            mapError: '맵 데이터를 불러오지 못했습니다.',
          });
        }
      },

      loadZones: async (robotId) => {
        try {
          const zones = await fetchRobotZones(robotId);
          set({ zones });
        } catch (error) {
          console.error('구역 목록 조회 실패:', error);
        }
      },

      loadZoneAirQuality: async (robotId) => {
        try {
          const zoneAirQuality = await fetchZoneAirQuality(robotId);
          set({
            zoneAirQuality,
            airQualityUpdatedAt: new Date().toISOString(),
            airQualityError: null,
          });
        } catch (error) {
          console.error('구역별 공기질 조회 실패:', error);
          set({ airQualityError: '공기질 데이터를 불러오지 못했습니다.' });
        }
      },

      updateZone: (zone) => set((state) => {
        const exists = state.zones.some((item) => item.id === zone.id);

        return {
          zones: exists
            ? state.zones.map((item) => (item.id === zone.id ? zone : item))
            : [...state.zones, zone],
        };
      }),

      saveZones: async (robotId) => {
        await saveRobotZones(robotId, get().zones);
      },

      clearLogs: () => set({ logs: [] }),
    }),
    {
      name: 'aria-robot-storage',
      partialize: (state) => ({
        isAiMode: state.isAiMode,
        isRunning: state.isRunning,
        robotStatus: state.robotStatus,
        battery: state.battery,
        airQuality: state.airQuality,
        robotStatusSummary: state.robotStatusSummary,
        robotPosition: state.robotPosition,
        logs: state.logs,
        mapData: state.mapData,
        zones: state.zones,
        zoneAirQuality: state.zoneAirQuality,
        airQualityUpdatedAt: state.airQualityUpdatedAt,
      }),
    }
  )
);

export default useRobotStore;

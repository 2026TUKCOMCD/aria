import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  fetchRobotEvents,
  fetchRobotDock,
  fetchRobotMap,
  fetchRobotStatus,
  fetchRobotZones,
  fetchZoneAirQuality,
  saveRobotDock,
  saveRobotZones,
  type RobotDockLocation,
  type RobotMap,
  type RobotStatusSummary,
  type RobotZone,
  type ZoneAirQuality,
  type ZonePoint,
} from '../api/ARIARobotController';

export type SetupStep =
  | 'MAP_REQUIRED'
  | 'DOCK_REQUIRED'
  | 'DOCK_MOVE_REQUIRED'
  | 'DOCK_CONFIRM_REQUIRED'
  | 'COMPLETE';

interface Log {
  time: string;
  content: string;
}

export interface SleepSchedule {
  wake_time: string;
  sleep_time: string;
  enabled: boolean;
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
  chargerPosition: ZonePoint | null;
  airQualityUpdatedAt: string | null;
  airQualityError: string | null;
  isChargerSetupComplete: boolean;
  isChargerSetupRequired: boolean;
  setupStep: SetupStep;
  sleepSchedule: SleepSchedule | null;
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
  loadChargerPosition: (robotId?: string) => Promise<void>;
  updateZone: (zone: RobotZone) => void;
  removeZone: (zoneId: number) => void;
  setChargerPosition: (position: ZonePoint | null) => void;
  confirmChargerSetup: () => void;
  resetChargerSetup: () => void;
  beginSetupFlow: () => void;
  setSetupStep: (step: SetupStep) => void;
  setSleepSchedule: (schedule: SleepSchedule | null) => void;
  saveChargerPosition: (robotId?: string) => Promise<void>;
  saveZones: (robotId?: string) => Promise<void>;
}

const formatLogTime = (value: string) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '--:--';

  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
};

const formatEventTime = (value?: string | number) => {
  if (!value) return formatLogTime(new Date().toISOString());

  if (typeof value === 'number') {
    return formatLogTime(new Date(value < 10_000_000_000 ? value * 1000 : value).toISOString());
  }

  return formatLogTime(value);
};

const getEventMessage = (type?: string, message?: string) => {
  if (message) return message;

  switch (type) {
    case 'POLLUTION_COOKING':
      return '요리로 인한 오염이 감지되었습니다.';
    case 'POLLUTION_NORMAL':
      return '일반 오염이 감지되었습니다.';
    case 'ACTIVITY_DETECTED':
      return '사람의 활동이 감지되었습니다.';
    case 'INACTIVITY_DETECTED':
      return '일정 시간 비활동 상태입니다.';
    case 'PURIFYING_START':
      return '공기 청정을 시작합니다.';
    case 'PURIFYING_DONE':
      return '청정이 완료되었습니다.';
    case 'PATROL_START':
      return '자율 순찰을 시작합니다.';
    default:
      return '이벤트가 발생했습니다.';
  }
};

const useRobotStore = create<RobotState>()(
  persist(
    (set, get) => ({
      isAiMode: false,
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
      chargerPosition: null,
      airQualityUpdatedAt: null,
      airQualityError: null,
      isChargerSetupComplete: false,
      isChargerSetupRequired: false,
      setupStep: 'MAP_REQUIRED',
      sleepSchedule: null,
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
          const statusValue = (
            robotStatusSummary.robot_status.current_zone ||
            ''
          ).toUpperCase();
          const isActuallyRunning =
            statusValue === 'RUNNING' ||
            statusValue === 'MOVING' ||
            statusValue === 'CLEANING';
          const shouldUpdateRunning = Boolean(statusValue);

          set({
            robotStatusSummary,
            robotStatusError: null,
            battery: robotStatusSummary.robot_status.battery,
            airQuality: {
              pm25: robotStatusSummary.air_quality.sensors.pm25,
              voc: robotStatusSummary.air_quality.sensors.voc,
            },
            ...(shouldUpdateRunning ? { isRunning: isActuallyRunning } : {}),
            ...(robotStatusSummary.pose
              ? {
                  robotPosition: {
                    robot_id: String(robotId || '1'),
                    x: robotStatusSummary.pose.x,
                    y: robotStatusSummary.pose.y,
                    theta: robotStatusSummary.pose.theta,
                    updated_at: new Date().toISOString(),
                  },
                }
              : {}),
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
          const events = await fetchRobotEvents(robotId);
          const formattedLogs = events.map((row) => ({
            time: formatEventTime(row.created_at || row.timestamp),
            content: getEventMessage(row.type || row.event || row.event_type, row.message),
          }));

          set({ logs: formattedLogs });
        } catch (error) {
          console.error('이벤트 로그 조회 실패:', error);
        }
      },

      loadMapData: async (robotId) => {
        set({ isMapLoading: true, mapError: null });

        try {
          const mapData = await fetchRobotMap(robotId);
          set({
            mapData,
            zones: mapData.zones && mapData.zones.length > 0 ? mapData.zones : get().zones,
            setupStep: get().setupStep === 'MAP_REQUIRED' ? 'DOCK_REQUIRED' : get().setupStep,
            isChargerSetupRequired: get().setupStep === 'MAP_REQUIRED' ? true : get().isChargerSetupRequired,
            isChargerSetupComplete: get().setupStep === 'MAP_REQUIRED' ? false : get().isChargerSetupComplete,
            isMapLoading: false,
            mapError: null,
          });
        } catch (error) {
          console.error('맵 데이터 조회 실패:', error);
          set({
            mapData: null,
            zones: [],
            zoneAirQuality: [],
            chargerPosition: null,
            isChargerSetupComplete: false,
            isChargerSetupRequired: true,
            setupStep: 'MAP_REQUIRED',
            isMapLoading: false,
            mapError: '맵 데이터를 불러오지 못했습니다.',
          });
        }
      },

      loadZones: async (robotId) => {
        try {
          const zones = await fetchRobotZones(robotId);
          if (zones.length === 0) return;
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

      loadChargerPosition: async (robotId) => {
        try {
          if (get().setupStep !== 'COMPLETE') return;

          const dock = await fetchRobotDock(robotId);
          if (!dock) return;

          set({
            chargerPosition: {
              x: dock.x,
              y: dock.y,
            },
            isChargerSetupComplete: true,
            isChargerSetupRequired: false,
            setupStep: 'COMPLETE',
          });
        } catch (error) {
          console.error('충전기 위치 조회 실패:', error);
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

      removeZone: (zoneId) => set((state) => ({
        zones: state.zones.filter((zone) => zone.id !== zoneId),
        zoneAirQuality: state.zoneAirQuality.filter((item) => item.zone_id !== zoneId),
      })),

      setChargerPosition: (position) => set({
        chargerPosition: position,
        isChargerSetupComplete: false,
      }),

      confirmChargerSetup: () => set({
        isChargerSetupComplete: true,
        isChargerSetupRequired: false,
        setupStep: 'COMPLETE',
      }),

      resetChargerSetup: () => set({
        chargerPosition: null,
        isChargerSetupComplete: false,
        isChargerSetupRequired: true,
        setupStep: 'DOCK_REQUIRED',
      }),

      beginSetupFlow: () => set({
        mapData: null,
        zones: [],
        chargerPosition: null,
        isChargerSetupComplete: false,
        isChargerSetupRequired: true,
        setupStep: 'MAP_REQUIRED',
      }),

      setSetupStep: (step) => set({
        setupStep: step,
        isChargerSetupComplete: step === 'COMPLETE',
        isChargerSetupRequired: step !== 'COMPLETE',
      }),

      setSleepSchedule: (schedule) => set({ sleepSchedule: schedule }),

      saveChargerPosition: async (robotId) => {
        const chargerPosition = get().chargerPosition;
        if (!chargerPosition) return;

        const payload: RobotDockLocation = {
          x: chargerPosition.x,
          y: chargerPosition.y,
          theta: 0,
        };

        await saveRobotDock(robotId, payload);
      },

      saveZones: async (robotId) => {
        const state = get();
        const mapZones = state.mapData?.zones || [];
        const sourceZones = mapZones.length > 0 ? mapZones : state.zones;
        const savedNamesById = new Map(state.zones.map((zone) => [zone.id, zone.name]));
        const zonesToSave = sourceZones.map((zone) => ({
          ...zone,
          name: savedNamesById.get(zone.id) || zone.name,
        }));

        await saveRobotZones(robotId, zonesToSave);
      },

      clearLogs: () => set({ logs: [] }),
    }),
    {
      name: 'aria-robot-storage',
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<RobotState>;

        return {
          ...currentState,
          ...persisted,
          mapData: null,
          zones: [],
          zoneAirQuality: [],
          chargerPosition: null,
          robotPosition: null,
        };
      },
      partialize: (state) => ({
        battery: state.battery,
        airQuality: state.airQuality,
        robotStatusSummary: state.robotStatusSummary,
        logs: state.logs,
        isChargerSetupComplete: state.isChargerSetupComplete,
        isChargerSetupRequired: state.isChargerSetupRequired,
        setupStep: state.setupStep,
        sleepSchedule: state.sleepSchedule,
        airQualityUpdatedAt: state.airQualityUpdatedAt,
      }),
    }
  )
);

export default useRobotStore;

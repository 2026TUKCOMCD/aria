import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AuthState {
  isLoggedIn: boolean;
  robotId: string | null;
  userName: string | null;
  robotName: string | null;
  qrToken: string | null;
  login: (payload: { robotId: string; userName: string; robotName: string; qrToken: string }) => void;
  logout: () => void;
}

const DEFAULT_ROBOT_ID = import.meta.env.VITE_ROBOT_ID || '1';

const normalizeRobotId = (robotId?: string | null) => {
  const value = String(robotId || '').trim();
  if (!value) return DEFAULT_ROBOT_ID;

  const normalized = value.toLowerCase();
  if (normalized === 'unknown' || normalized === 'undefined' || normalized === 'null') {
    return DEFAULT_ROBOT_ID;
  }

  return value;
};

const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      isLoggedIn: false,
      robotId: null,
      userName: null,
      robotName: null,
      qrToken: null,
      login: ({ robotId, userName, robotName, qrToken }) => set({
        isLoggedIn: true,
        robotId: normalizeRobotId(robotId),
        userName,
        robotName,
        qrToken,
      }),
      logout: () => set({
        isLoggedIn: false,
        robotId: null,
        userName: null,
        robotName: null,
        qrToken: null,
      }),
    }),
    {
      name: 'aria-auth-storage',
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<AuthState>;

        return {
          ...currentState,
          ...persisted,
          robotId: normalizeRobotId(persisted.robotId),
        };
      },
    }
  )
);

export default useAuthStore;

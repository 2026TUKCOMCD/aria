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
        robotId,
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
    }
  )
);

export default useAuthStore;

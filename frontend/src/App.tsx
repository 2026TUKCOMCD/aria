import { useEffect } from 'react';
import { BrowserRouter as Router, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { io, type Socket } from 'socket.io-client';

import MainPage from './pages/MainPage';
import AuthPage from './pages/AuthPage';
import ErrorPage from './pages/ErrorPage';
import MapPage from './pages/MapPage';
import SettingsPage from './pages/SettingsPage';

import Navigation from './components/Navigation';
import useAuthStore from './store/useAuthStore';
import useRobotStore from './store/useRobotStore';

const SOCKET_SERVER_URL = import.meta.env.VITE_ARIA_API_URL || 'http://localhost:3000';

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const isLoggedIn = useAuthStore((state) => state.isLoggedIn);
  const location = useLocation();

  if (!isLoggedIn) {
    return <Navigate to="/auth" replace state={{ from: location.pathname }} />;
  }

  return children;
};

const AppShell = () => {
  const location = useLocation();
  const isLoggedIn = useAuthStore((state) => state.isLoggedIn);
  const { addLog, setIsRunning, setRobotPosition } = useRobotStore();
  const shouldShowNavigation = isLoggedIn && location.pathname !== '/auth';

  useEffect(() => {
    let socket: Socket | null = null;

    if (isLoggedIn) {
      socket = io(SOCKET_SERVER_URL, {
        transports: ['websocket'],
        reconnectionAttempts: 5,
      });

      socket.on('connect', () => {
        console.log('WebSocket 서버에 연결되었습니다. ID:', socket?.id);
      });

      socket.on('robot_alert', (data) => {
        const logMessage = data.message || '알 수 없는 이벤트 발생';
        addLog(logMessage);

        if (data.event_type === 'EMERGENCY') {
          alert(`[긴급] ${logMessage}`);
        }
      });

      socket.on('status_change', (newStatus) => {
        if (newStatus === 'RUNNING') setIsRunning(true);
        else if (newStatus === 'IDLE') setIsRunning(false);
      });

      socket.on('robot_position', (position) => {
        if (
          position &&
          typeof position.x === 'number' &&
          typeof position.y === 'number' &&
          typeof position.theta === 'number'
        ) {
          setRobotPosition({
            robot_id: String(position.robot_id || ''),
            x: position.x,
            y: position.y,
            theta: position.theta,
            updated_at: position.updated_at || new Date().toISOString(),
          });
        }
      });

      socket.on('connect_error', (err) => {
        console.error('소켓 연결 에러:', err.message);
      });
    }

    return () => {
      socket?.disconnect();
    };
  }, [addLog, isLoggedIn, setIsRunning, setRobotPosition]);

  return (
    <div className="min-h-screen bg-gray-200 flex justify-center items-center">
      <div
        className="relative flex min-h-screen w-full max-w-[450px] flex-col overflow-hidden font-sans shadow-2xl"
        style={{
          background: 'radial-gradient(circle at 50% 0%, #ffffff 30%, #e0ebff 100%)',
        }}
      >
        <div className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/auth" element={isLoggedIn ? <Navigate to="/" replace /> : <AuthPage />} />
            <Route path="/" element={<ProtectedRoute><MainPage /></ProtectedRoute>} />
            <Route path="/map" element={<ProtectedRoute><MapPage /></ProtectedRoute>} />
            <Route path="/settings" element={<ProtectedRoute><SettingsPage /></ProtectedRoute>} />
            <Route path="/error" element={<ErrorPage />} />
            <Route path="*" element={<Navigate to={isLoggedIn ? '/' : '/auth'} replace />} />
          </Routes>
        </div>

        {shouldShowNavigation && <Navigation />}
      </div>
    </div>
  );
};

function App() {
  return (
    <Router>
      <AppShell />
    </Router>
  );
}

export default App;

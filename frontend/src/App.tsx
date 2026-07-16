import { useEffect } from 'react';
import { BrowserRouter as Router, Navigate, Route, Routes, useLocation } from 'react-router-dom';

import MainPage from './pages/MainPage';
import AuthPage from './pages/AuthPage';
import ErrorPage from './pages/ErrorPage';
import MapPage from './pages/MapPage';
import SettingsPage from './pages/SettingsPage';

import Navigation from './components/Navigation';
import useAuthStore from './store/useAuthStore';
import useRobotStore from './store/useRobotStore';

const EVENT_STREAM_URL =
  import.meta.env.VITE_ARIA_SOCKET_URL ||
  import.meta.env.VITE_ARIA_API_URL ||
  'http://localhost:3000';

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const isLoggedIn = useAuthStore((state) => state.isLoggedIn);
  const mapData = useRobotStore((state) => state.mapData);
  const isChargerSetupComplete = useRobotStore((state) => state.isChargerSetupComplete);
  const isChargerSetupRequired = useRobotStore((state) => state.isChargerSetupRequired);
  const location = useLocation();

  if (!isLoggedIn) {
    return <Navigate to="/auth" replace state={{ from: location.pathname }} />;
  }

  if (mapData && (!isChargerSetupComplete || isChargerSetupRequired) && location.pathname !== '/map') {
    return <Navigate to="/map" replace state={{ requireChargerSetup: true }} />;
  }

  return children;
};

const AppShell = () => {
  const location = useLocation();
  const isLoggedIn = useAuthStore((state) => state.isLoggedIn);
  const authRobotId = useAuthStore((state) => state.robotId);
  const qrToken = useAuthStore((state) => state.qrToken);
  const robotId = authRobotId || import.meta.env.VITE_ROBOT_ID || '1';
  const { addLog, setIsRunning, setRobotPosition } = useRobotStore();
  const shouldShowNavigation = isLoggedIn && !location.pathname.startsWith('/auth');

  useEffect(() => {
    if (!isLoggedIn || !qrToken) return;

    const eventSource = new EventSource(
      `${EVENT_STREAM_URL}/robots/${robotId}/events/stream?token=${encodeURIComponent(qrToken)}`
    );

    const handleAlertEvent = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        const eventType = data.event || data.type || data.event_type;
        const status = data.status;
        const logMessage =
          data.message ||
          (eventType === 'clean_status' && status === 'COMPLETE'
            ? '청정이 완료되었습니다.'
            : eventType === 'lwt_status' || status === 'Offline'
              ? '로봇 연결이 비정상적으로 종료되었습니다.'
              : '이벤트가 발생했습니다.');

        addLog(logMessage);

        if (eventType === 'clean_status' && status === 'COMPLETE') {
          setIsRunning(false);
          alert(logMessage);
        }

        if (eventType === 'lwt_status' || status === 'Offline' || data.type === 'EMERGENCY' || data.event_type === 'EMERGENCY') {
          setIsRunning(false);
          alert(logMessage);
        }
      } catch (error) {
        console.error('SSE 이벤트 파싱 실패:', error);
      }
    };

    eventSource.addEventListener('connected', handleAlertEvent);
    eventSource.addEventListener('clean_status', handleAlertEvent);
    eventSource.addEventListener('lwt_status', handleAlertEvent);
    eventSource.addEventListener('robot_alert', handleAlertEvent);

    eventSource.addEventListener('status_change', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status === 'RUNNING' || data.type === 'RUNNING') setIsRunning(true);
        if (data.status === 'IDLE' || data.type === 'IDLE') setIsRunning(false);
      } catch (error) {
        console.error('SSE 상태 이벤트 파싱 실패:', error);
      }
    });

    eventSource.addEventListener('robot_position', (event) => {
      try {
        const position = JSON.parse(event.data);
        if (
          position &&
          typeof position.x === 'number' &&
          typeof position.y === 'number' &&
          typeof position.theta === 'number'
        ) {
          setRobotPosition({
            robot_id: String(position.robot_id || robotId),
            x: position.x,
            y: position.y,
            theta: position.theta,
            updated_at: position.updated_at || new Date().toISOString(),
          });
        }
      } catch (error) {
        console.error('SSE 위치 이벤트 파싱 실패:', error);
      }
    });

    eventSource.onerror = () => {
      console.error('SSE 연결 오류');
    };

    return () => {
      eventSource.close();
    };
  }, [addLog, isLoggedIn, qrToken, robotId, setIsRunning, setRobotPosition]);

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
            <Route path="/auth/qr" element={<AuthPage />} />
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

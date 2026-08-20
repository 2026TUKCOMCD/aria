import { useEffect } from 'react';
import { BrowserRouter as Router, Navigate, Route, Routes, useLocation } from 'react-router-dom';

import MainPage from './pages/MainPage';
import AuthPage from './pages/AuthPage';
import ErrorPage from './pages/ErrorPage';
import MapPage from './pages/MapPage';
import SettingsPage from './pages/SettingsPage';

import Navigation from './components/Navigation';
import { sendRobotCommand } from './api/ARIARobotController';
import useAuthStore from './store/useAuthStore';
import useRobotStore from './store/useRobotStore';

const EVENT_STREAM_URL =
  import.meta.env.VITE_ARIA_SOCKET_URL ||
  import.meta.env.VITE_ARIA_API_URL ||
  'http://localhost:3000';

const BYPASS_SETUP_GUARD_FOR_TEST = false;

const formatTimeHHmm = (date: Date) =>
  `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;

const formatScheduleKeyDate = (date: Date) =>
  `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;

const getSafeRobotId = (robotId?: string | null) => {
  const value = String(robotId || '').trim();
  if (!value) return import.meta.env.VITE_ROBOT_ID || '1';

  const normalized = value.toLowerCase();
  if (normalized === 'unknown' || normalized === 'undefined' || normalized === 'null') {
    return import.meta.env.VITE_ROBOT_ID || '1';
  }

  return value;
};

const getRobotEventMessage = (type?: string, fallback?: string) => {
  if (fallback) return fallback;

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

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const isLoggedIn = useAuthStore((state) => state.isLoggedIn);
  const setupStep = useRobotStore((state) => state.setupStep);
  const location = useLocation();

  if (!isLoggedIn) {
    return <Navigate to="/auth" replace state={{ from: location.pathname }} />;
  }

  if (!BYPASS_SETUP_GUARD_FOR_TEST && setupStep === 'MAP_REQUIRED' && location.pathname !== '/map') {
    return <Navigate to="/map" replace state={{ requireChargerSetup: true }} />;
  }

  return children;
};

const AppShell = () => {
  const location = useLocation();
  const isLoggedIn = useAuthStore((state) => state.isLoggedIn);
  const authRobotId = useAuthStore((state) => state.robotId);
  const qrToken = useAuthStore((state) => state.qrToken);
  const robotId = getSafeRobotId(authRobotId);
  const { addLog, setIsRunning, setRobotPosition, setupStep, setSetupStep, sleepSchedule } = useRobotStore();
  const shouldShowNavigation =
    isLoggedIn && setupStep !== 'MAP_REQUIRED' && !location.pathname.startsWith('/auth');

  useEffect(() => {
    if (!isLoggedIn || !qrToken) return;

    const eventSource = new EventSource(
      `${EVENT_STREAM_URL}/robots/${robotId}/events/stream?token=${encodeURIComponent(qrToken)}`
    );

    const handleAlertEvent = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        const eventType = data.event || data.type || data.event_type;
        const type = String(data.type || data.event || data.event_type || '').trim().toUpperCase();
        const status = data.status;
        const isPurifyingStart = type === 'PURIFYING_START';
        const isPurifyingDone =
          type === 'PURIFYING_DONE' ||
          type === 'CLEAN_DONE' ||
          type === 'CLEAN_STATUS' ||
          data.type === 'CLEAN_DONE' ||
          (eventType === 'clean_status' && status === 'COMPLETE');
        const isMoveComplete =
          setupStep === 'DOCK_MOVE_REQUIRED' &&
          (
            status === 'COMPLETE' ||
            status === 'DONE' ||
            status === 'IDLE' ||
            status === 'ARRIVED' ||
            data.type === 'DOCK_MOVE_DONE' ||
            data.type === 'MOVE_DONE' ||
            data.type === 'NAVIGATION_DONE' ||
            data.type === 'ARRIVED' ||
            eventType === 'move_status' ||
            eventType === 'navigate_status' ||
            eventType === 'navigation_status'
          );
        const logMessage =
          eventType === 'lwt_status' || status === 'Offline'
            ? '로봇 연결이 비정상적으로 종료되었습니다.'
            : getRobotEventMessage(type, data.message);

        addLog(logMessage);

        if (isPurifyingStart) {
          setIsRunning(true);
          return;
        }

        if (isMoveComplete) {
          setIsRunning(false);
          setSetupStep('DOCK_CONFIRM_REQUIRED');
          return;
        }

        if (isPurifyingDone) {
          setIsRunning(false);
          alert(data.message || '청정이 완료되었습니다.');
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
    eventSource.addEventListener('message', handleAlertEvent);

    eventSource.addEventListener('status_change', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status === 'RUNNING' || data.type === 'RUNNING') setIsRunning(true);
        if (data.status === 'IDLE' || data.type === 'IDLE') {
          setIsRunning(false);
          if (setupStep === 'DOCK_MOVE_REQUIRED') {
            setSetupStep('DOCK_CONFIRM_REQUIRED');
          }
        }
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
  }, [addLog, isLoggedIn, qrToken, robotId, setIsRunning, setRobotPosition, setSetupStep, setupStep]);

  useEffect(() => {
    if (!isLoggedIn || !sleepSchedule?.enabled) return;

    let lastSentKey = '';

    const syncScheduledMode = async () => {
      const now = new Date();
      const currentTime = formatTimeHHmm(now);
      const mode =
        currentTime === sleepSchedule.sleep_time
          ? 'LOW'
          : currentTime === sleepSchedule.wake_time
            ? 'WAIT'
            : null;

      if (!mode) return;

      const sendKey = `${formatScheduleKeyDate(now)}-${currentTime}-${mode}`;
      if (lastSentKey === sendKey) return;
      lastSentKey = sendKey;

      try {
        await sendRobotCommand(robotId, 'MODE', mode);
        addLog(
          mode === 'LOW'
            ? '취침 시간이 되어 저전력 모드로 전환했습니다.'
            : '기상 시간이 되어 대기 모드로 전환했습니다.'
        );

        if (mode === 'LOW') {
          setIsRunning(false);
        }
      } catch (error) {
        lastSentKey = '';
        console.error('예약 모드 전환 실패:', error);
      }
    };

    syncScheduledMode();
    const timerId = window.setInterval(syncScheduledMode, 30000);

    return () => {
      window.clearInterval(timerId);
    };
  }, [addLog, isLoggedIn, robotId, setIsRunning, sleepSchedule]);

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

import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { io, Socket } from 'socket.io-client';

// 페이지 컴포넌트
import MainPage from './pages/MainPage'; 
import AuthPage from './pages/AuthPage';
import ErrorPage from './pages/ErrorPage';
import MapPage from './pages/MapPage';
import SettingsPage from './pages/SettingsPage';

// 공통 컴포넌트 및 스토어
import Navigation from './components/Navigation';
import useRobotStore from './store/useRobotStore';

// WebSocket 서버 주소 (성국님의 Node.js 서버 주소로 수정하세요)
const SOCKET_SERVER_URL = "http://localhost:3000"; 

function App() {
  const { addLog, isRunning, setIsRunning } = useRobotStore();
  
  // 실제 구현 시에는 AuthStore 등에서 로그인 상태를 가져와야 함.
  // 임시로 true로 설정하거나 로컬스토리지 토큰 존재 여부로 판단.
  const isLoggedIn = true; 

  useEffect(() => {
    let socket: Socket | null = null;

    if (isLoggedIn) {
      // 1. 소켓 연결 생성
      socket = io(SOCKET_SERVER_URL, {
        transports: ['websocket'], // 성능을 위해 웹소켓 우선 사용
        reconnectionAttempts: 5,    // 연결 실패 시 5번 재시도
      });

      // 2. 연결 성공 이벤트
      socket.on('connect', () => {
        console.log('✅ WebSocket 서버에 연결되었습니다. ID:', socket?.id);
      });

      // 3. 실시간 로봇 알림 수신 (Node.js의 io.emit('robot_alert', ...)와 매칭)
      socket.on('robot_alert', (data) => {
        console.log('🔔 실시간 알림 도착:', data);
        
        // Zustand 스토어에 로그 기록 추가
        // data 구조: { robot_id: string, event_type: string, message: string }
        const logMessage = data.message || "알 수 없는 이벤트 발생";
        addLog(logMessage);

        // 예: 특정 이벤트 타입에 따라 추가 액션 수행
        if (data.event_type === 'EMERGENCY') {
          alert(`[긴급] ${logMessage}`);
        }
      });

      // 4. 로봇 상태 동기화 (옵션: 서버에서 상태 변경 시 알려준다면)
      socket.on('status_change', (newStatus) => {
        if (newStatus === 'RUNNING') setIsRunning(true);
        else if (newStatus === 'IDLE') setIsRunning(false);
      });

      // 5. 연결 에러 처리
      socket.on('connect_error', (err) => {
        console.error('❌ 소켓 연결 에러:', err.message);
      });
    }

    // [Cleanup] 컴포넌트 언마운트 또는 로그아웃 시 소켓 연결 해제
    return () => {
      if (socket) {
        console.log('🔌 소켓 연결을 해제합니다.');
        socket.disconnect();
      }
    };
  }, [isLoggedIn, addLog, setIsRunning]);

  return (
    <Router>
      <div className="min-h-screen bg-gray-200 flex justify-center items-center">
        {/* 모바일 뷰 컨테이너 */}
        <div 
          className="relative w-full max-w-[450px] min-h-screen shadow-2xl flex flex-col overflow-hidden font-sans"
          style={{ 
            background: 'radial-gradient(circle at 50% 0%, #ffffff 30%, #e0ebff 100%)' 
          }}
        >
          {/* 메인 콘텐츠 영역 */}
          <div className="flex-1 overflow-y-auto">
            <Routes>
              <Route path="/" element={<MainPage />} />
              <Route path="/map" element={<MapPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/auth" element={<AuthPage />} />
              <Route path="/error" element={<ErrorPage />} />
            </Routes>
          </div>
          
          {/* 하단 네비게이션바 (고정) */}
          <Navigation />
        </div>
      </div>
    </Router>
  );
}

export default App;
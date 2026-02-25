import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Main from './pages/Main'; // 성국 님이 작성한 Main.tsx 경로

function App() {
  return (
    <Router>
      <Routes>
        {/* 기본 경로(/)에 Main 컴포넌트를 배치합니다 */}
        <Route path="/" element={<Main />} />
        
        {/* 나중에 추가할 페이지들 예시 */}
        <Route path="/map" element={<div>맵 페이지</div>} />
        <Route path="/settings" element={<div>설정 페이지</div>} />
      </Routes>
    </Router>
  );
}

export default App;
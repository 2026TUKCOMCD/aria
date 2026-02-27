import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage'; 
import AuthPage from './pages/AuthPage';
import ErrorPage from './pages/ErrorPage';
import Navigation from './components/Navigation';
import MapPage from './pages/MapPage';
import SettingsPage from './pages/SettingsPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-200 flex justify-center items-center">
        <div 
          className="relative w-full max-w-[450px] min-h-screen shadow-2xl flex flex-col overflow-hidden font-sans"
          style={{ 
            background: 'radial-gradient(circle at 50% 0%, #ffffff 30%, #e0ebff 100%)' 
          }}
        >
          <div className="flex-1 overflow-y-auto">
            <Routes>
              <Route path="/" element={<MainPage />} />
              <Route path="/map" element={<MapPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/auth" element={<AuthPage />} />
              <Route path="/error" element={<ErrorPage />} />
            </Routes>
          </div>
          
          <Navigation />
        </div>
      </div>
    </Router>
  );
}

export default App;
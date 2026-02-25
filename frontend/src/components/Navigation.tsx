import { useNavigate, useLocation } from 'react-router-dom';
import HomeIcon from '../assets/home.svg?react';
import MapIcon from '../assets/map.svg?react';
import SettingIcon from '../assets/settng.svg?react';

// 아이콘 타입을 정의합니다. (React 컴포넌트 형태)
type IconComponent = React.FunctionComponent<React.SVGProps<SVGSVGElement>>;

interface NavItem {
  name: string;
  path: string;
  Icon: IconComponent;
}

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  const navItems: NavItem[] = [
    { name: '맵', path: '/map', Icon: MapIcon as IconComponent },
    { name: '기기', path: '/', Icon: HomeIcon as IconComponent },
    { name: '설정', path: '/settings', Icon: SettingIcon as IconComponent },
  ];

  return (
    <nav className="fixed bottom-0 left-0 right-0 h-[80px] bg-white border-t border-gray-100 flex items-center justify-around px-6 z-[50]">
      {navItems.map((item) => {
        const active = isActive(item.path);
        return (
          <button
            key={item.name}
            onClick={() => navigate(item.path)}
            className={`flex flex-col items-center justify-center gap-1 transition-colors ${
              active ? 'text-black' : 'text-gray-400'
            }`}
          >
            {/* div로 감싸서 크기를 조절하면 타입 에러가 발생하지 않습니다. */}
            <div className="w-7 h-7 flex items-center justify-center">
              <item.Icon />
            </div>
            <span className="text-[14px] font-[700]">{item.name}</span>
          </button>
        );
      })}
    </nav>
  );
};

export default Navigation;
import { useLocation, useNavigate } from 'react-router-dom';
import HomeIcon from '../assets/home.svg?react';
import MapIcon from '../assets/map.svg?react';
import SettingIcon from '../assets/settng.svg?react';

type IconComponent = React.FunctionComponent<React.SVGProps<SVGSVGElement>>;

interface NavItem {
  name: string;
  path: string;
  Icon: IconComponent;
}

const navItems: NavItem[] = [
  { name: '맵', path: '/map', Icon: MapIcon as IconComponent },
  { name: '기기', path: '/', Icon: HomeIcon as IconComponent },
  { name: '설정', path: '/settings', Icon: SettingIcon as IconComponent },
];

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="fixed bottom-0 left-1/2 z-[50] flex h-[80px] w-full max-w-[450px] -translate-x-1/2 items-center justify-around border-t border-gray-100 bg-white px-6">
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
            <div className="flex h-7 w-7 items-center justify-center">
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

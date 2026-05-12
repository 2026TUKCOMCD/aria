import AIIcon from '../assets/ai.svg?react';
import ActivityIcon from '../assets/activity.svg?react';
import CookingIcon from '../assets/cooking.svg?react';
import PatrolIcon from '../assets/patrol.svg?react';
import SleepIcon from '../assets/sleep.svg?react';

interface AIModeManualModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const manualItems = [
  {
    title: '공기질 감지',
    description: 'PM2.5와 VOC 값을 기준으로 공기질 등급을 판단합니다.',
    Icon: CookingIcon,
  },
  {
    title: '이벤트 판단',
    description: '요리 오염, 활동/비활동, 취침 시간 같은 상황을 이벤트로 기록합니다.',
    Icon: ActivityIcon,
  },
  {
    title: '자동 이동',
    description: '오염 구역이나 설정된 구역을 기준으로 청정 위치를 결정합니다.',
    Icon: PatrolIcon,
  },
  {
    title: '스케줄 반영',
    description: '취침 시간에는 대기 상태로 전환하고 기상 시간에 정상 동작합니다.',
    Icon: SleepIcon,
  },
];

const AIModeManualModal = ({ isOpen, onClose }: AIModeManualModalProps) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/40 px-6">
      <div className="relative w-full max-w-[390px] rounded-[28px] bg-white p-6 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute right-5 top-5 p-2 text-main-blue transition-transform hover:scale-110"
          aria-label="닫기"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>

        <div className="mb-6 flex items-center gap-3 pr-10">
          <div className="flex h-14 w-14 items-center justify-center rounded-[18px] bg-main-blue text-white">
            <AIIcon className="h-8 w-8" />
          </div>
          <div>
            <h2 className="text-[28px] font-black text-gray-900">AI 모드</h2>
            <p className="text-[14px] font-bold text-gray-400">자동 청정 동작 안내</p>
          </div>
        </div>

        <div className="space-y-3">
          {manualItems.map(({ title, description, Icon }) => (
            <div key={title} className="grid grid-cols-[48px_1fr] gap-3 rounded-[18px] bg-main-sky p-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-[15px] bg-white text-main-blue shadow-sm">
                <Icon className="h-7 w-7" />
              </div>
              <div>
                <h3 className="text-[17px] font-black text-gray-900">{title}</h3>
                <p className="mt-1 text-[13px] font-bold leading-5 text-gray-500">{description}</p>
              </div>
            </div>
          ))}
        </div>

        <button
          onClick={onClose}
          className="mt-6 h-[56px] w-full rounded-[18px] bg-main-blue text-[20px] font-black text-white shadow-lg transition-all active:scale-95"
        >
          확인
        </button>
      </div>
    </div>
  );
};

export default AIModeManualModal;

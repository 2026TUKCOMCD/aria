import { useEffect, useMemo, useState } from 'react';
import CheckIcon from '../assets/check.svg?react';
import MapIcon from '../assets/map.svg?react';
import AIIcon from '../assets/ai.svg?react';
import PinIcon from '../assets/pin.svg?react';
import Navigation from '../components/Navigation';

import Activity from '../assets/activity.svg?react';
import NonActivity from '../assets/non_activity.svg?react';
import Cooking from '../assets/cooking.svg?react';
import NonCooking from '../assets/non_cooking.svg?react';
import Sleep from '../assets/sleep.svg?react';
import Morning from '../assets/morning.svg?react';
import Return from '../assets/return.svg?react';
import Home from '../assets/home.svg?react';
import Patrol from '../assets/patrol.svg?react';

import EventLogModal from '../components/EventLogModal';
import AIModeManualModal from '../components/AIModeManualModal';
import { navigateRobot, sendRobotCommand } from '../api/ARIARobotController';
import useRobotStore from '../store/useRobotStore';
import useAuthStore from '../store/useAuthStore';

const formatUpdatedAt = (value?: string) => {
  if (!value) return '갱신 정보 없음';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString('ko-KR', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const airGradeConfig = {
  GOOD: { label: '좋음', className: 'bg-emerald-100 text-emerald-700' },
  NORMAL: { label: '보통', className: 'bg-amber-100 text-amber-700' },
  BAD: { label: '나쁨', className: 'bg-red-100 text-red-700' },
  CRITICAL: { label: '위험', className: 'bg-red-600 text-white' },
};

const MainPage = () => {
  const {
    logs,
    isRunning,
    setIsRunning,
    battery,
    isAiMode,
    setAiMode,
    fetchLogs,
    mapData,
    zones,
    isMapLoading,
    mapError,
    loadMapData,
    loadZones,
    loadRobotStatus,
    robotStatusSummary,
    robotStatusError,
  } = useRobotStore();

  const [isLogOpen, setIsLogOpen] = useState(false);
  const [isManualOpen, setIsManualOpen] = useState(false);
  const [selectedZoneId, setSelectedZoneId] = useState<number | null>(null);

  const authRobotId = useAuthStore((state) => state.robotId);
  const robotId = import.meta.env.VITE_ROBOT_ID || authRobotId || '1';
  const hasMapData = Boolean(mapData);
  const metadata = mapData?.metadata;

  const selectedZone = useMemo(
    () => zones.find((zone) => zone.id === selectedZoneId) || zones[0] || null,
    [selectedZoneId, zones]
  );

  const worldSize = useMemo(() => {
    if (!metadata) return null;

    return {
      width: metadata.width * metadata.resolution,
      height: metadata.height * metadata.resolution,
    };
  }, [metadata]);

  const worldToPercent = (point: { x: number; y: number }) => {
    if (!metadata || !worldSize) return { left: 50, top: 50 };

    const [originX, originY] = metadata.origin;
    const rawLeft = ((point.x - originX) / worldSize.width) * 100;
    const rawTop = (1 - (point.y - originY) / worldSize.height) * 100;

    return {
      left: Math.min(94, Math.max(6, rawLeft)),
      top: Math.min(94, Math.max(6, rawTop)),
    };
  };

  useEffect(() => {
    loadMapData(robotId);
    loadZones(robotId);
    loadRobotStatus(robotId);
  }, [loadMapData, loadRobotStatus, loadZones, robotId]);

  useEffect(() => {
    if (!selectedZoneId && zones.length > 0) {
      setSelectedZoneId(zones[0].id);
    }
  }, [selectedZoneId, zones]);

  const handleOpenLogs = async () => {
    setIsLogOpen(true);
    await fetchLogs(robotId);
  };

  const renderStatusIcon = (message: string) => {
    const iconClass = 'h-7 w-7 fill-current';

    if (message.includes('요리 오염')) return <Cooking className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('비요리 오염')) return <NonCooking className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('비활동')) return <NonActivity className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('활동')) return <Activity className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('취침')) return <Sleep className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('기상')) return <Morning className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('복귀중')) return <Return className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('복귀 완료')) return <Home className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('순찰')) return <Patrol className={`${iconClass} text-main-blue animate-pulse`} />;

    return <div className="h-3 w-3 animate-pulse rounded-full bg-main-red" />;
  };

  const displayStatusMessage = () => {
    if (logs.length > 0) return logs[0].content;
    if (isRunning && isAiMode) return 'AI 자동 청정 중';
    if (isRunning && selectedZone) return `${selectedZone.name}으로 이동 중`;
    return isAiMode ? 'AI가 공기질을 감지합니다.' : '이동할 구역을 선택해주세요.';
  };

  const handleModeSelect = (targetMode: 'BASIC' | 'AI') => {
    if (isRunning) {
      alert('로봇이 작동 중일 때는 모드를 변경할 수 없습니다. 먼저 중지해주세요.');
      return;
    }

    setAiMode(targetMode === 'AI');
  };

  const handleActionClick = async () => {
    if (!hasMapData) {
      try {
        await sendRobotCommand(robotId, 'SLAM', 'ON');
        alert('맵 데이터 생성을 시작합니다. 생성 후 맵 새로고침을 눌러주세요.');
      } catch (error) {
        console.error('맵 생성 시작 실패:', error);
        alert('맵 생성 명령을 전달하지 못했습니다.');
      }
      return;
    }

    try {
      if (isRunning) {
        await sendRobotCommand(robotId, 'POWER', 'OFF');
        setIsRunning(false);
        return;
      }

      if (!isAiMode) {
        if (!selectedZone) {
          alert('먼저 이동할 구역을 설정해주세요.');
          return;
        }

        await sendRobotCommand(robotId, 'MODE', 'MANUAL');
        await navigateRobot(robotId, { type: 'ZONE', zone_id: selectedZone.id });
        setIsRunning(true);
        return;
      }

      await sendRobotCommand(robotId, 'MODE', 'AUTO');
      await sendRobotCommand(robotId, 'POWER', 'ON');
      setIsRunning(true);
    } catch (error) {
      console.error('로봇 제어 실패:', error);
      alert('로봇에게 명령을 전달하지 못했습니다.');
    }
  };

  return (
    <div className="flex min-h-screen flex-col pb-[100px] font-sans">
      <header className="flex items-center justify-between px-9 pt-10">
        <h1 className="text-[20px] font-[900] tracking-tight text-black">ARIA</h1>
        <span className="text-[18px] font-bold text-main-blue">배터리: {battery}%</span>
      </header>

      <section className="mt-3 px-6">
        <div className={`w-full rounded-[30px] border p-3 shadow-xl transition-all ${
          isRunning ? 'border-gray-300 bg-gray-200' : 'border-main-sky bg-main-sky'
        }`}>
          <div className="flex h-[65px] w-full items-center rounded-[25px] bg-white p-1.5">
            <button
              onClick={() => handleModeSelect('BASIC')}
              className={`flex h-full flex-1 items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
                !isAiMode ? 'bg-main-blue text-white shadow-md' : 'text-gray-400'
              } ${isRunning && isAiMode ? 'cursor-not-allowed opacity-50' : ''}`}
            >
              기본 모드
              {!isAiMode && <CheckIcon className="h-5 w-5 fill-current" />}
            </button>
            <button
              onClick={() => handleModeSelect('AI')}
              className={`flex h-full flex-1 items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
                isAiMode ? 'bg-main-blue text-white shadow-md' : 'text-gray-400'
              } ${isRunning && !isAiMode ? 'cursor-not-allowed opacity-50' : ''}`}
            >
              AI 모드
              {isAiMode && <CheckIcon className="h-5 w-5 fill-current" />}
            </button>
          </div>
          {isAiMode && (
            <button
              onClick={() => setIsManualOpen(true)}
              className="mt-3 h-[38px] w-full rounded-[16px] bg-white text-[14px] font-black text-main-blue shadow-inner transition-all active:scale-[0.98]"
            >
              AI 모드 안내 보기
            </button>
          )}
        </div>
      </section>

      <section className="mt-3 px-6">
        <div className="flex flex-col gap-2 rounded-[25px] bg-main-sky p-3 shadow-xl">
          <div className="flex min-h-[40px] items-center gap-3 px-2">
            {isRunning ? (
              <>
                {renderStatusIcon(displayStatusMessage())}
                <span className="text-[20px] font-black text-main-blue">{displayStatusMessage()}</span>
              </>
            ) : hasMapData ? (
              <>
                {isAiMode ? (
                  <AIIcon className="h-7 w-7 text-main-blue" />
                ) : (
                  <PinIcon className="h-7 w-7 text-main-blue" />
                )}
                <span className="text-[20px] font-black text-main-blue">{displayStatusMessage()}</span>
              </>
            ) : (
              <>
                <MapIcon className="h-7 w-7 text-main-blue" />
                <span className="text-[20px] font-black text-main-blue">맵 데이터를 생성해주세요</span>
              </>
            )}
          </div>
          <button
            onClick={handleOpenLogs}
            className="mt-1 w-full rounded-[40px] bg-white px-4 py-1 text-left text-[16px] font-extrabold text-black shadow-inner transition-transform active:scale-[0.98]"
          >
            이벤트 로그 보기 &gt;
          </button>
        </div>
      </section>

      {robotStatusSummary && (
        <section className="mt-3 px-6">
          <div className="grid grid-cols-[120px_1fr] gap-3 rounded-[25px] bg-white p-4 shadow-lg">
            <div className="flex flex-col items-center justify-center rounded-[18px] bg-main-sky py-3">
              <span className="text-[13px] font-black text-main-blue">공기질 점수</span>
              <span className="text-[38px] font-black leading-none text-main-blue">
                {robotStatusSummary.air_quality.score}
              </span>
              <span className={`mt-2 rounded-full px-3 py-1 text-[12px] font-black ${
                airGradeConfig[robotStatusSummary.air_quality.grade].className
              }`}>
                {airGradeConfig[robotStatusSummary.air_quality.grade].label}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-[14px] bg-gray-50 px-3 py-2">
                <p className="text-[11px] font-bold text-gray-400">PM2.5</p>
                <p className="text-[17px] font-black text-gray-800">
                  {robotStatusSummary.air_quality.sensors.pm25}<span className="text-[11px]"> µg/m³</span>
                </p>
              </div>
              <div className="rounded-[14px] bg-gray-50 px-3 py-2">
                <p className="text-[11px] font-bold text-gray-400">VOC</p>
                <p className="text-[17px] font-black text-gray-800">{robotStatusSummary.air_quality.sensors.voc}</p>
              </div>
              <div className="rounded-[14px] bg-gray-50 px-3 py-2">
                <p className="text-[11px] font-bold text-gray-400">온도</p>
                <p className="text-[17px] font-black text-gray-800">{robotStatusSummary.air_quality.sensors.temperature}°C</p>
              </div>
              <div className="rounded-[14px] bg-gray-50 px-3 py-2">
                <p className="text-[11px] font-bold text-gray-400">습도</p>
                <p className="text-[17px] font-black text-gray-800">{robotStatusSummary.air_quality.sensors.humidity}%</p>
              </div>
            </div>

            <div className="col-span-2 flex items-center justify-between rounded-[14px] bg-gray-50 px-3 py-2">
              <span className="text-[12px] font-bold text-gray-500">
                전원 {robotStatusSummary.robot_status.power} · 모드 {robotStatusSummary.robot_status.mode}
              </span>
              <span className="text-[12px] font-bold text-gray-500">
                {robotStatusSummary.robot_status.is_charging ? '충전 중' : '배터리 사용 중'}
              </span>
            </div>
          </div>
        </section>
      )}

      {robotStatusError && (
        <section className="mt-3 px-6">
          <p className="rounded-[16px] bg-main-red/10 px-4 py-3 text-center text-[13px] font-bold text-main-red">
            {robotStatusError}
          </p>
        </section>
      )}

      <section className="mt-3 flex flex-1 px-6">
        <div className="relative flex w-full flex-col overflow-hidden rounded-[30px] border-4 border-gray-200 bg-white shadow-lg">
          {isMapLoading && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4 bg-white/80">
              <div className="h-14 w-14 animate-spin rounded-full border-8 border-main-blue border-t-transparent" />
              <span className="font-black text-main-blue">맵 확인 중</span>
            </div>
          )}

          {hasMapData && mapData ? (
            <div className="flex h-full flex-col gap-3 p-4">
              <div className="flex items-center justify-end">
                <span className="text-[12px] font-bold text-gray-400">{formatUpdatedAt(mapData.last_updated)}</span>
              </div>

              <div
                className="relative w-full overflow-hidden rounded-[20px] bg-gray-50"
                style={{
                  aspectRatio: `${mapData.metadata.width} / ${mapData.metadata.height}`,
                }}
              >
                <img
                  src={mapData.map_url}
                  alt={mapData.map_name}
                  className="absolute inset-0 h-full w-full object-fill"
                  draggable={false}
                />
                {!isAiMode && zones.map((zone) => {
                  const position = worldToPercent(zone.center);
                  const active = selectedZone?.id === zone.id;

                  return (
                    <button
                      key={`map-label-${zone.id}`}
                      type="button"
                      onClick={() => setSelectedZoneId(zone.id)}
                      className={`absolute z-10 max-w-[120px] -translate-x-1/2 -translate-y-1/2 truncate rounded-full px-3 py-1.5 text-[12px] font-black shadow-md transition-all active:scale-95 ${
                        active
                          ? 'bg-main-blue text-white ring-2 ring-white'
                          : 'bg-white/90 text-main-blue ring-1 ring-main-blue/20'
                      }`}
                      style={{
                        left: `${position.left}%`,
                        top: `${position.top}%`,
                      }}
                    >
                      {zone.name}
                    </button>
                  );
                })}
              </div>

              {!isAiMode && (
                <div className="grid grid-cols-2 gap-2">
                  {zones.length > 0 ? zones.map((zone) => (
                    <button
                      key={zone.id}
                      onClick={() => setSelectedZoneId(zone.id)}
                      className={`h-[42px] rounded-[14px] text-[15px] font-black shadow-sm transition-all ${
                        selectedZone?.id === zone.id
                          ? 'bg-main-blue text-white'
                          : 'bg-main-sky text-main-blue'
                      }`}
                    >
                      {zone.name}
                    </button>
                  )) : (
                    <div className="col-span-2 rounded-[14px] bg-gray-100 px-4 py-3 text-center text-[14px] font-bold text-gray-400">
                      맵 페이지에서 구역을 설정해주세요
                    </div>
                  )}
                </div>
              )}

              {mapError && (
                <p className="rounded-[14px] bg-main-red/10 px-4 py-2 text-center text-[13px] font-bold text-main-red">
                  {mapError}
                </p>
              )}
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-4">
              <span className="text-[20px] font-black text-black opacity-30">No Map Data</span>
              <button
                onClick={() => loadMapData(robotId)}
                className="rounded-[16px] bg-main-sky px-5 py-3 text-[15px] font-black text-main-blue"
              >
                맵 새로고침
              </button>
            </div>
          )}
        </div>
      </section>

      <section className="mt-3 mb-1 px-6">
        <button
          onClick={handleActionClick}
          className={`flex h-[60px] w-full items-center justify-center gap-3 rounded-[20px] text-[20px] font-black shadow-lg transition-all active:scale-95 ${
            isRunning
              ? 'border-2 border-main-blue bg-white text-main-blue'
              : 'bg-main-blue text-white'
          }`}
        >
          {!hasMapData ? '맵 데이터 생성' : isRunning ? '청정 중지' : isAiMode ? 'AI 청정 시작' : '선택 구역 이동'}
        </button>
      </section>

      <Navigation />

      <EventLogModal
        isOpen={isLogOpen}
        onClose={() => setIsLogOpen(false)}
        logs={logs}
      />
      <AIModeManualModal
        isOpen={isManualOpen}
        onClose={() => setIsManualOpen(false)}
      />
    </div>
  );
};

export default MainPage;

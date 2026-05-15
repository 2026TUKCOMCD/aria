import { Fragment, useEffect, useMemo, useState, type MouseEvent } from 'react';
import CheckIcon from '../assets/check.svg?react';
import MapIcon from '../assets/map.svg?react';
import Navigation from '../components/Navigation';
import NameInputModal from '../components/NameInputModal';
import useRobotStore from '../store/useRobotStore';
import useAuthStore from '../store/useAuthStore';
import type { AirQualityStatus, RobotZone, ZoneAirQuality, ZoneArea, ZonePoint } from '../api/ARIARobotController';

const AIR_QUALITY_STALE_MINUTES = 5;

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

const getEffectiveAirQualityStatus = (item?: ZoneAirQuality): AirQualityStatus => {
  if (!item || !item.measured_at) return 'STALE';

  const measuredAt = new Date(item.measured_at).getTime();
  if (Number.isNaN(measuredAt)) return 'STALE';

  const diffMinutes = (Date.now() - measuredAt) / 60000;
  if (diffMinutes > AIR_QUALITY_STALE_MINUTES) return 'STALE';

  return item.status;
};

const airQualityStyle: Record<AirQualityStatus, { label: string; className: string }> = {
  GOOD: { label: '좋음', className: 'border-emerald-500 bg-emerald-400/35 text-emerald-700' },
  NORMAL: { label: '보통', className: 'border-amber-500 bg-amber-300/40 text-amber-700' },
  BAD: { label: '나쁨', className: 'border-main-red bg-main-red/30 text-main-red' },
  STALE: { label: '지연', className: 'border-gray-400 bg-gray-300/45 text-gray-600' },
};

const MapPage = () => {
  const [tab, setTab] = useState<'AREA' | 'LOCATION'>('AREA');
  const [selectedZoneId, setSelectedZoneId] = useState<number | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const {
    mapData,
    zones,
    zoneAirQuality,
    airQualityUpdatedAt,
    airQualityError,
    robotPosition,
    isMapLoading,
    mapError,
    loadMapData,
    loadZones,
    loadZoneAirQuality,
    updateZone,
    removeZone,
    saveZones,
  } = useRobotStore();

  const authRobotId = useAuthStore((state) => state.robotId);
  const robotId = authRobotId || import.meta.env.VITE_ROBOT_ID || '1';
  const metadata = mapData?.metadata;

  const selectedZone = useMemo(
    () => zones.find((zone) => zone.id === selectedZoneId) || null,
    [selectedZoneId, zones]
  );

  const airQualityByZoneId = useMemo(() => {
    return new Map(zoneAirQuality.map((item) => [item.zone_id, item]));
  }, [zoneAirQuality]);

  useEffect(() => {
    loadMapData(robotId);
    loadZones(robotId);
    loadZoneAirQuality(robotId);
  }, [loadMapData, loadZoneAirQuality, loadZones, robotId]);

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
    const left = ((point.x - originX) / worldSize.width) * 100;
    const top = (1 - (point.y - originY) / worldSize.height) * 100;

    return { left, top };
  };

  const areaToStyle = (area?: ZoneArea) => {
    if (!area || !metadata || !worldSize) return null;

    const [originX, originY] = metadata.origin;
    const left = ((area.x_min - originX) / worldSize.width) * 100;
    const top = (1 - (area.y_max - originY) / worldSize.height) * 100;
    const width = ((area.x_max - area.x_min) / worldSize.width) * 100;
    const height = ((area.y_max - area.y_min) / worldSize.height) * 100;

    return {
      left: `${left}%`,
      top: `${top}%`,
      width: `${width}%`,
      height: `${height}%`,
    };
  };

  const polygonToPoints = (polygon?: ZonePoint[]) => {
    if (!polygon || polygon.length < 3 || !metadata || !worldSize) return '';

    return polygon
      .map((point) => {
        const position = worldToPercent(point);
        return `${position.left},${position.top}`;
      })
      .join(' ');
  };

  const handleMapClick = (event: MouseEvent<HTMLDivElement>) => {
    if (tab !== 'LOCATION' || !metadata || !worldSize) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const clickXPercent = ((event.clientX - rect.left) / rect.width) * 100;
    const clickYPercent = ((event.clientY - rect.top) / rect.height) * 100;
    const [originX, originY] = metadata.origin;

    const x = originX + (clickXPercent / 100) * worldSize.width;
    const y = originY + (1 - clickYPercent / 100) * worldSize.height;
    const nextId = zones.length > 0 ? Math.max(...zones.map((zone) => zone.id)) + 1 : 1;

    const newZone: RobotZone = {
      id: nextId,
      name: `새 구역 ${nextId}`,
      center: {
        x: Number(x.toFixed(2)),
        y: Number(y.toFixed(2)),
      },
    };

    updateZone(newZone);
    setSelectedZoneId(newZone.id);
  };

  const handleSaveZoneName = (newName: string) => {
    if (!selectedZone) return;

    updateZone({ ...selectedZone, name: newName });
    setSelectedZoneId(null);
  };

  const handleDeleteSelectedZone = () => {
    if (!selectedZone) return;

    const shouldDelete = window.confirm(`"${selectedZone.name}" 구역을 삭제할까요?`);
    if (!shouldDelete) return;

    removeZone(selectedZone.id);
    setSelectedZoneId(null);
  };

  const handleSaveZones = async () => {
    setIsSaving(true);

    try {
      await saveZones(robotId);
      alert('구역 정보가 저장되었습니다.');
    } catch (error) {
      console.error('구역 정보 저장 실패:', error);
      alert('구역 정보를 저장하지 못했습니다.');
    } finally {
      setIsSaving(false);
    }
  };

  const robotPoint = robotPosition ? worldToPercent(robotPosition) : null;

  return (
    <div className="flex min-h-screen flex-col pb-[100px] font-sans">
      <header className="flex flex-col items-center justify-center pt-12">
        <h2 className="text-[20px] font-black tracking-tight text-main-blue">
          {tab === 'AREA' ? '구역을 눌러서 이름을 설정해주세요' : '맵을 눌러 위치를 설정해주세요'}
        </h2>
        <p className="mt-2 text-[14px] font-bold text-gray-500">
          마지막 갱신: {formatUpdatedAt(mapData?.last_updated)}
        </p>
      </header>

      <section className="mt-3 px-6">
        <div className="w-full rounded-[30px] border border-main-sky bg-main-sky p-3 shadow-xl">
          <div className="flex h-[65px] w-full items-center rounded-[25px] bg-white p-1.5">
            <button
              onClick={() => setTab('AREA')}
              className={`flex h-full flex-1 items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
                tab === 'AREA' ? 'bg-main-blue text-white shadow-md' : 'text-gray-400'
              }`}
            >
              구역이름
              {tab === 'AREA' && <CheckIcon className="h-5 w-5 fill-current" />}
            </button>

            <button
              onClick={() => setTab('LOCATION')}
              className={`flex h-full flex-1 items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
                tab === 'LOCATION' ? 'bg-main-blue text-white shadow-md' : 'text-gray-400'
              }`}
            >
              위치설정
              {tab === 'LOCATION' && <CheckIcon className="h-5 w-5 fill-current" />}
            </button>
          </div>
        </div>
      </section>

      <section className="mt-3 flex flex-1 px-6">
        <div className="relative flex w-full flex-col items-center justify-center overflow-hidden rounded-[30px] border-4 border-gray-200 bg-white shadow-xl">
          {isMapLoading && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4 bg-white/80">
              <div className="h-14 w-14 animate-spin rounded-full border-8 border-main-blue border-t-transparent" />
              <span className="font-black text-main-blue">맵 데이터를 불러오는 중입니다</span>
            </div>
          )}

          {mapData ? (
            <div className="w-full p-4">
              <div
                className="relative w-full overflow-hidden rounded-[20px] bg-gray-50"
                style={{
                  aspectRatio: metadata ? `${metadata.width} / ${metadata.height}` : '4 / 3',
                }}
                onClick={handleMapClick}
              >
                <img
                  src={mapData.map_url}
                  alt={mapData.map_name}
                  className="absolute inset-0 h-full w-full object-fill"
                  draggable={false}
                />

                {zones.map((zone) => {
                  const point = worldToPercent(zone.center);
                  const areaStyle = areaToStyle(zone.area);
                  const polygonPoints = polygonToPoints(zone.polygon);
                  const hasPolygon = Boolean(polygonPoints);
                  const isSelected = selectedZoneId === zone.id;
                  const status = getEffectiveAirQualityStatus(airQualityByZoneId.get(zone.id));

                  return (
                    <Fragment key={zone.id}>
                      {hasPolygon && (
                        <svg
                          className="pointer-events-none absolute inset-0 h-full w-full"
                          viewBox="0 0 100 100"
                          preserveAspectRatio="none"
                        >
                          <polygon
                            points={polygonPoints}
                            className={`${airQualityStyle[status].className} ${isSelected ? 'stroke-main-red' : ''}`}
                            fill="currentColor"
                            fillOpacity="0.18"
                            stroke={isSelected ? 'currentColor' : zone.color || 'currentColor'}
                            strokeWidth="0.6"
                            vectorEffect="non-scaling-stroke"
                          />
                        </svg>
                      )}

                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelectedZoneId(zone.id);
                        }}
                        className="absolute z-10 -translate-x-1/2 -translate-y-1/2"
                        style={{ left: `${point.left}%`, top: `${point.top}%` }}
                      >
                        {!hasPolygon && areaStyle && (
                          <>
                            <span
                              className={`pointer-events-none absolute border-2 ${airQualityStyle[status].className}`}
                              style={{
                                ...areaStyle,
                                left: `calc(${areaStyle.left} - ${point.left}%)`,
                                top: `calc(${areaStyle.top} - ${point.top}%)`,
                              }}
                            />
                            <span
                              className={`pointer-events-none absolute border-2 ${
                                isSelected ? 'border-main-red' : 'border-main-blue/60'
                              }`}
                              style={{
                                ...areaStyle,
                                left: `calc(${areaStyle.left} - ${point.left}%)`,
                                top: `calc(${areaStyle.top} - ${point.top}%)`,
                              }}
                            />
                          </>
                        )}
                        <span
                          className={`flex min-h-8 min-w-8 items-center justify-center rounded-full px-2 text-[12px] font-black text-white shadow-lg ${
                            isSelected ? 'bg-main-red' : 'bg-main-blue'
                          }`}
                        >
                          {zone.name}
                        </span>
                        <span className="mt-1 block rounded-full bg-white/90 px-2 py-0.5 text-[10px] font-black text-gray-600 shadow">
                          {airQualityStyle[status].label}
                        </span>
                      </button>
                    </Fragment>
                  );
                })}

                {robotPoint && robotPosition && (
                  <div
                    className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-1/2"
                    style={{ left: `${robotPoint.left}%`, top: `${robotPoint.top}%` }}
                  >
                    <div
                      className="relative flex h-10 w-10 items-center justify-center rounded-full border-4 border-white bg-main-red shadow-xl"
                      style={{ transform: `rotate(${robotPosition.theta}rad)` }}
                    >
                      <span className="absolute -top-3 h-4 w-2 rounded-full bg-main-red" />
                      <span className="h-3 w-3 rounded-full bg-white" />
                    </div>
                    <div className="mt-1 rounded-full bg-white/95 px-2 py-0.5 text-center text-[10px] font-black text-main-red shadow">
                      ARIA
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-3 rounded-[16px] bg-white px-3 py-3 shadow-inner">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[13px] font-black text-gray-700">공기질 오버레이</span>
                  <span className="text-[11px] font-bold text-gray-400">
                    {formatUpdatedAt(airQualityUpdatedAt || undefined)}
                  </span>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {(['GOOD', 'NORMAL', 'BAD', 'STALE'] as AirQualityStatus[]).map((status) => (
                    <div key={status} className="flex items-center justify-center gap-1">
                      <span className={`h-3 w-3 rounded-full border ${airQualityStyle[status].className}`} />
                      <span className="text-[11px] font-bold text-gray-600">{airQualityStyle[status].label}</span>
                    </div>
                  ))}
                </div>
                {robotPosition && (
                  <p className="mt-2 text-center text-[11px] font-bold text-gray-500">
                    로봇 위치: x {robotPosition.x.toFixed(2)}, y {robotPosition.y.toFixed(2)}
                  </p>
                )}
                {airQualityError && (
                  <p className="mt-2 text-center text-[11px] font-bold text-main-red">{airQualityError}</p>
                )}
              </div>

              {mapError && (
                <p className="mt-3 rounded-[14px] bg-main-red/10 px-4 py-3 text-center text-[14px] font-bold text-main-red">
                  {mapError} 이전 맵을 유지합니다.
                </p>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <MapIcon className="h-20 w-20 text-gray-200 opacity-50" />
              <span className="font-bold text-gray-400">
                {mapError || '저장된 맵 데이터가 없습니다'}
              </span>
            </div>
          )}
        </div>
      </section>

      <section className="mt-3 mb-1 grid grid-cols-[1fr_1fr] gap-3 px-6">
        <button
          onClick={() => {
            loadMapData(robotId);
            loadZones(robotId);
            loadZoneAirQuality(robotId);
          }}
          className="flex h-[60px] items-center justify-center rounded-[20px] border-2 border-main-blue bg-white text-[18px] font-black text-main-blue shadow-lg transition-all active:scale-95"
        >
          맵 새로고침
        </button>
        <button
          onClick={handleSaveZones}
          disabled={isSaving}
          className="flex h-[60px] items-center justify-center rounded-[20px] bg-main-blue text-[18px] font-black text-white shadow-lg transition-all active:scale-95 disabled:bg-gray-400"
        >
          {isSaving ? '저장 중' : '현재 상태 저장'}
        </button>
      </section>

      {selectedZone && (
        <section className="mb-3 px-6">
          <button
            onClick={handleDeleteSelectedZone}
            className="flex h-[52px] w-full items-center justify-center rounded-[18px] border-2 border-main-red bg-white text-[16px] font-black text-main-red shadow-md transition-all active:scale-95"
          >
            선택 구역 삭제
          </button>
        </section>
      )}

      <NameInputModal
        isOpen={tab === 'AREA' && Boolean(selectedZone)}
        currentName={selectedZone?.name || ''}
        onClose={() => setSelectedZoneId(null)}
        onSave={handleSaveZoneName}
      />

      <Navigation />
    </div>
  );
};

export default MapPage;

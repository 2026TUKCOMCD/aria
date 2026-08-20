import axios from 'axios';

export interface MapMetadata {
  resolution: number;
  origin: [number, number, number];
  width: number;
  height: number;
}

export interface RobotMap {
  robot_id: string;
  map_name: string;
  map_url: string;
  metadata: MapMetadata;
  zones?: RobotZone[];
  last_updated: string;
}

interface RobotMapResponse {
  robot_id: string;
  map_name?: string;
  map_url?: string;
  image_url?: string;
  metadata: MapMetadata;
  zones?: RobotZoneResponse[];
  last_updated?: string;
}

export interface ZoneArea {
  x_min?: number;
  y_min?: number;
  x_max?: number;
  y_max?: number;
  width?: number;
  height?: number;
}

export interface ZonePoint {
  x: number;
  y: number;
}

export interface RobotZone {
  id: number;
  name: string;
  center: ZonePoint;
  color?: string;
  area?: ZoneArea;
  polygon?: ZonePoint[];
  air_score?: number;
  air_grade?: RobotAirQualityGrade | 'STALE';
}

type ZonePointResponse = ZonePoint | [number, number] | string;

interface RobotZoneResponse {
  id: number;
  name: string;
  center?: ZonePointResponse;
  color?: string;
  area?: ZoneArea;
  polygon?: ZonePointResponse[] | string;
  air_score?: number;
  air_grade?: string;
}

interface SaveRobotZonePayload {
  id: number;
  name: string;
  center: ZonePoint;
  area?: ZoneArea;
  polygon?: [number, number][];
}

export type AirQualityStatus = 'GOOD' | 'NORMAL' | 'BAD' | 'STALE';
type RobotAirQualityGrade = 'GOOD' | 'NORMAL' | 'BAD' | 'CRITICAL';

export interface ZoneAirQuality {
  zone_id: number;
  pm25: number | null;
  voc: number | null;
  status: AirQualityStatus;
  measured_at: string | null;
}

export interface RobotStatusSummary {
  robot_status: {
    battery: number;
    is_charging: boolean;
    power: 'ON' | 'OFF' | 'SLEEP' | 'OFFLINE' | string;
    mode: 'AUTO' | 'MANUAL' | 'TURBO' | 'WAIT' | 'LOW' | string;
    current_zone: string | null;
  };
  air_quality: {
    score: number;
    grade: RobotAirQualityGrade;
    sensors: {
      pm25: number;
      voc: number;
      temperature: number;
      humidity: number;
    };
  };
  pose?: {
    x: number;
    y: number;
    theta: number;
  };
  last_updated?: string;
}

export interface AuthVerifyResult {
  valid: boolean;
  robot_id?: string;
  user_name?: string;
  robot_name?: string;
  message?: string;
}

interface AuthVerifyResponse {
  valid?: boolean;
  isAuthorized?: boolean;
  robot_id?: string;
  user_name?: string;
  robot_name?: string;
  context?: {
    robot_id?: string;
    user_name?: string;
    robot_name?: string;
  };
  message?: string;
}

export interface StoreQrTokenPayload {
  qr_token: string;
  robot_id: string;
  user_name?: string;
  robot_name?: string;
}

export interface StoreQrTokenResult {
  message?: string;
  qr_token?: string;
  error?: string;
}

export interface RobotSchedulePayload {
  wake_time: string;
  sleep_time: string;
  enabled: boolean;
}

export interface RobotEventLog {
  log_id?: number;
  type?: string;
  event?: string;
  event_type?: string;
  message?: string;
  timestamp?: number | string;
  created_at?: string;
}

export interface RobotDockLocation {
  x: number;
  y: number;
  theta?: number;
  updated_at?: string;
}

const API_BASE_URL = import.meta.env.VITE_ARIA_API_URL || 'http://localhost:3000';
const SOCKET_BASE_URL =
  import.meta.env.VITE_ARIA_SOCKET_URL ||
  import.meta.env.VITE_ARIA_API_URL ||
  'http://localhost:3000';
const API_TOKEN = import.meta.env.VITE_API_SECRET_TOKEN;
const DEFAULT_ID = import.meta.env.VITE_ROBOT_ID || '1';
const REQUEST_TIMEOUT_MS = 8000;

const getRobotId = (robotId?: string | null) => {
  const value = String(robotId || '').trim();
  if (!value) return DEFAULT_ID;

  const normalized = value.toLowerCase();
  if (normalized === 'unknown' || normalized === 'undefined' || normalized === 'null') {
    return DEFAULT_ID;
  }

  return value;
};

const getStoredQrToken = () => {
  try {
    const rawAuth = window.localStorage.getItem('aria-auth-storage');
    if (!rawAuth) return null;

    const parsedAuth = JSON.parse(rawAuth);
    return parsedAuth?.state?.qrToken || null;
  } catch {
    return null;
  }
};

const createAuthHeaders = () => {
  const qrToken = getStoredQrToken();

  return {
    'Content-Type': 'application/json',
    ...(API_TOKEN ? { 'X-ARIA-SECRET': API_TOKEN } : {}),
    ...(qrToken ? { Authorization: qrToken } : {}),
  };
};

const createTokenHeaders = (token: string) => ({
  Authorization: token,
});

const parseJsonString = <T>(value: string): T | null => {
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
};

const normalizeZonePoint = (point: ZonePointResponse | undefined): ZonePoint => {
  if (!point) return { x: 0, y: 0 };

  if (typeof point === 'string') {
    const parsed = parseJsonString<ZonePointResponse>(point);
    return normalizeZonePoint(parsed || undefined);
  }

  if (Array.isArray(point)) {
    return { x: point[0], y: point[1] };
  }

  return point;
};

const normalizeZonePolygon = (polygon?: ZonePointResponse[] | string): ZonePoint[] | undefined => {
  if (!polygon) return undefined;

  if (typeof polygon === 'string') {
    const parsed = parseJsonString<ZonePointResponse[]>(polygon);
    return normalizeZonePolygon(parsed || undefined);
  }

  return polygon.map(normalizeZonePoint);
};

const normalizeZoneAirGrade = (grade: unknown): RobotZone['air_grade'] | undefined => {
  const normalized = String(grade || '').trim().toUpperCase();
  if (
    normalized === 'GOOD' ||
    normalized === 'NORMAL' ||
    normalized === 'BAD' ||
    normalized === 'CRITICAL' ||
    normalized === 'STALE'
  ) {
    return normalized;
  }

  return undefined;
};

const normalizeZone = (zone: RobotZoneResponse): RobotZone => ({
  id: zone.id,
  name: zone.name,
  center: normalizeZonePoint(zone.center),
  color: zone.color,
  area: zone.area,
  polygon: normalizeZonePolygon(zone.polygon),
  air_score: typeof zone.air_score === 'number' ? zone.air_score : undefined,
  air_grade: normalizeZoneAirGrade(zone.air_grade),
});

const normalizeZones = (zones?: RobotZoneResponse[]): RobotZone[] => {
  if (!Array.isArray(zones)) return [];
  return zones.map(normalizeZone);
};

const normalizeAirQualityGrade = (grade: unknown): RobotAirQualityGrade => {
  const normalized = String(grade || '').trim().toUpperCase();
  if (normalized === 'GOOD' || normalized === 'NORMAL' || normalized === 'BAD' || normalized === 'CRITICAL') {
    return normalized;
  }

  return 'NORMAL';
};

export const sendRobotCommand = async (robotId: string, target: string, action: string) => {
  const targetId = getRobotId(robotId);
  const command = target === 'SET_MODE' ? 'MODE' : target;
  const value = action;

  try {
    const response = await axios.post(
      `${API_BASE_URL}/robots/${targetId}/command`,
      { command, value },
      { headers: createAuthHeaders(), timeout: REQUEST_TIMEOUT_MS }
    );
    return response.data;
  } catch (error) {
    console.log('전송 시도 URL:', `${API_BASE_URL}/robots/${targetId}/command`);
    console.log('토큰 로드 여부:', API_TOKEN ? '성공' : '실패(undefined)');
    throw error;
  }
};

export const fetchRobotEvents = async (robotId?: string): Promise<RobotEventLog[]> => {
  const targetId = getRobotId(robotId);

  try {
    const response = await axios.get(`${API_BASE_URL}/robots/${targetId}/events`, {
      headers: createAuthHeaders(),
      timeout: REQUEST_TIMEOUT_MS,
    });

    if (Array.isArray(response.data)) return response.data;
    return response.data.data || response.data.events || [];
  } catch (error) {
    const response = await axios.get(`${SOCKET_BASE_URL}/api/events`, {
      params: { robot_id: targetId },
      timeout: REQUEST_TIMEOUT_MS,
    });

    return response.data.data || [];
  }
};

export const verifyQrToken = async (token: string): Promise<AuthVerifyResult> => {
  const response = await axios.get(`${API_BASE_URL}/auth/verify`, {
    headers: createTokenHeaders(token),
    timeout: REQUEST_TIMEOUT_MS,
    validateStatus: (status) => status >= 200 && status < 500,
  });
  const data = response.data as AuthVerifyResponse;

  return {
    valid: Boolean(data.valid ?? data.isAuthorized),
    robot_id: data.robot_id || data.context?.robot_id,
    user_name: data.user_name || data.context?.user_name,
    robot_name: data.robot_name || data.context?.robot_name,
    message: data.message,
  };
};

export const storeQrToken = async (
  payload: StoreQrTokenPayload
): Promise<StoreQrTokenResult> => {
  const response = await axios.post(`${API_BASE_URL}/auth/verify`, payload, {
    headers: createAuthHeaders(),
    timeout: REQUEST_TIMEOUT_MS,
    validateStatus: (status) => status >= 200 && status < 500,
  });
  return response.data;
};

export const fetchRobotMap = async (robotId?: string): Promise<RobotMap> => {
  const targetId = getRobotId(robotId);
  const response = await axios.get(`${API_BASE_URL}/robots/${targetId}/map`, {
    headers: createAuthHeaders(),
    timeout: REQUEST_TIMEOUT_MS,
  });
  const data = response.data as RobotMapResponse;
  const mapUrl = data.map_url || data.image_url;

  if (!mapUrl) {
    throw new Error('Map response must include map_url or image_url');
  }

  return {
    robot_id: data.robot_id || targetId,
    map_name: data.map_name || 'ARIA map',
    map_url: mapUrl,
    metadata: data.metadata,
    zones: normalizeZones(data.zones),
    last_updated: data.last_updated || new Date().toISOString(),
  };
};

export const fetchRobotZones = async (robotId?: string): Promise<RobotZone[]> => {
  const targetId = getRobotId(robotId);
  const response = await axios.get(`${API_BASE_URL}/robots/${targetId}/zones`, {
    headers: createAuthHeaders(),
    timeout: REQUEST_TIMEOUT_MS,
  });
  if (Array.isArray(response.data)) return normalizeZones(response.data);
  if (Array.isArray(response.data.zones)) return normalizeZones(response.data.zones);
  if (response.data?.id && response.data?.name) return [normalizeZone(response.data)];
  return [];
};

export const saveRobotZones = async (robotId: string | undefined, zones: RobotZone[]) => {
  const targetId = getRobotId(robotId);
  const payload: SaveRobotZonePayload[] = zones.map(({ id, name, center, area, polygon }) => ({
    id,
    name,
    center,
    area,
    polygon: polygon?.map((point) => [point.x, point.y]),
  }));
  console.log('구역 저장 요청:', {
    url: `${API_BASE_URL}/robots/${targetId}/zones`,
    payload: { zones: payload },
  });
  const response = await axios.put(
    `${API_BASE_URL}/robots/${targetId}/zones`,
    { zones: payload },
    { headers: createAuthHeaders(), timeout: REQUEST_TIMEOUT_MS }
  );
  console.log('구역 저장 응답:', response.status, response.data);
  return response.data;
};

export const fetchRobotDock = async (robotId?: string): Promise<RobotDockLocation | null> => {
  const targetId = getRobotId(robotId);
  const response = await axios.get(`${API_BASE_URL}/robots/${targetId}/dock`, {
    headers: createAuthHeaders(),
    timeout: REQUEST_TIMEOUT_MS,
    validateStatus: (status) => (status >= 200 && status < 300) || status === 404,
  });

  if (response.status === 404) return null;
  return response.data.data || response.data.dock || response.data.charger_position || response.data;
};

export const saveRobotDock = async (
  robotId: string | undefined,
  payload: RobotDockLocation
) => {
  const targetId = getRobotId(robotId);
  const response = await axios.post(`${API_BASE_URL}/robots/${targetId}/dock`, payload, {
    headers: createAuthHeaders(),
    timeout: REQUEST_TIMEOUT_MS,
  });
  return response.data;
};

export const fetchZoneAirQuality = async (_robotId?: string): Promise<ZoneAirQuality[]> => {
  return [];
};

export const fetchRobotStatus = async (robotId?: string): Promise<RobotStatusSummary> => {
  const targetId = getRobotId(robotId);
  const response = await axios.get(`${API_BASE_URL}/robots/${targetId}/status`, {
    headers: createAuthHeaders(),
    timeout: REQUEST_TIMEOUT_MS,
  });
  const rawData = response.data || {};
  const data = rawData.data || rawData.result || rawData;
  const robotStatus = data.robot_status || data.robotStatus || data.status || {};
  const airQuality = data.air_quality || data.airQuality || {};
  const legacyAirQuality = robotStatus.air_quality || robotStatus.airQuality || {};
  const sensors =
    data.sensors ||
    data.sensor ||
    airQuality.sensors ||
    robotStatus.sensors ||
    rawData.sensor ||
    rawData.sensors ||
    {};
  const pose = data.pose || data.robot_pose || rawData.pose || robotStatus.pose;
  const score =
    airQuality.score ??
    data.air_score ??
    data.score ??
    rawData.air_score ??
    rawData.score ??
    legacyAirQuality.score;
  const grade =
    airQuality.grade ??
    data.air_grade ??
    data.grade ??
    rawData.air_grade ??
    rawData.grade ??
    legacyAirQuality.grade;

  return {
    robot_status: {
      battery: Number(robotStatus.battery ?? 100),
      is_charging: Boolean(robotStatus.is_charging),
      power: robotStatus.power || 'OFF',
      mode: robotStatus.mode || 'MANUAL',
      current_zone: robotStatus.current_zone ?? null,
    },
    air_quality: {
      score: Number(score ?? 0),
      grade: normalizeAirQualityGrade(grade),
      sensors: {
        pm25: Number(sensors.pm25 ?? 0),
        voc: Number(sensors.voc ?? 0),
        temperature: Number(sensors.temperature ?? 0),
        humidity: Number(sensors.humidity ?? 0),
      },
    },
    pose: pose && typeof pose.x === 'number' && typeof pose.y === 'number'
      ? {
          x: pose.x,
          y: pose.y,
          theta: Number(pose.theta ?? 0),
        }
      : undefined,
    last_updated: data.last_updated || rawData.last_updated,
  };
};

export const saveRobotSchedule = async (
  robotId: string | undefined,
  payload: RobotSchedulePayload
) => {
  const targetId = getRobotId(robotId);
  const response = await axios.post(`${API_BASE_URL}/robots/${targetId}/schedule`, payload, {
    headers: createAuthHeaders(),
    timeout: REQUEST_TIMEOUT_MS,
  });
  return response.data;
};

export const resetRobotData = async (robotId: string | undefined, target: 'MAP' | 'AI') => {
  const targetId = getRobotId(robotId);
  const response = await axios.post(
    `${API_BASE_URL}/robots/${targetId}/reset`,
    { target },
    { headers: createAuthHeaders(), timeout: REQUEST_TIMEOUT_MS }
  );
  return response.data;
};

export const navigateRobot = async (
  robotId: string | undefined,
  payload:
    | { type: 'MOVE_TO'; x: number; y: number; theta?: number }
    | { type: 'MOVE_TO'; target_type: 'ZONE'; zone_id: number; zone_name: string; x: number; y: number; theta?: number }
    | { type: 'CANCEL_NAVIGATION' }
) => {
  const targetId = getRobotId(robotId);
  const url = `${API_BASE_URL}/robots/${targetId}/navigate`;
  const headers = createAuthHeaders();

  console.log('이동 명령 요청:', {
    url,
    payload,
    hasToken: Boolean(headers.Authorization),
  });

  const response = await axios.post(url, payload, {
    headers,
    timeout: REQUEST_TIMEOUT_MS,
  });

  console.log('이동 명령 응답:', response.status, response.data);
  return response.data;
};

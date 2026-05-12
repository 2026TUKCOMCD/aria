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
  last_updated: string;
}

export interface ZoneArea {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface RobotZone {
  id: number;
  name: string;
  center: {
    x: number;
    y: number;
  };
  area?: ZoneArea;
}

export type AirQualityStatus = 'GOOD' | 'NORMAL' | 'BAD' | 'STALE';

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
    power: 'ON' | 'OFF' | 'SLEEP';
    mode: 'AUTO' | 'MANUAL' | 'TURBO';
    current_zone: string | null;
  };
  air_quality: {
    score: number;
    grade: 'GOOD' | 'NORMAL' | 'BAD' | 'CRITICAL';
    sensors: {
      pm25: number;
      voc: number;
      temperature: number;
      humidity: number;
    };
  };
}

export interface AuthVerifyResult {
  valid: boolean;
  robot_id: string;
  user_name: string;
  robot_name: string;
}

export interface RobotSchedulePayload {
  wake_time: string;
  sleep_time: string;
  enabled: boolean;
}

const API_BASE_URL = import.meta.env.VITE_ARIA_API_URL || 'http://localhost:3000';
const API_TOKEN = import.meta.env.VITE_API_SECRET_TOKEN;
const DEFAULT_ID = import.meta.env.VITE_ROBOT_ID || '1';

const getRobotId = (robotId?: string) => robotId || DEFAULT_ID;

const authHeaders = {
  'Content-Type': 'application/json',
  ...(API_TOKEN ? { 'X-ARIA-SECRET': API_TOKEN } : {}),
};

const createTokenHeaders = (token: string) => ({
  'Content-Type': 'application/json',
  Authorization: `Bearer ${token}`,
  'X-ARIA-QR-TOKEN': token,
});

export const sendRobotCommand = async (robotId: string, target: string, action: string) => {
  const targetId = getRobotId(robotId);

  try {
    const response = await axios.post(
      `${API_BASE_URL}/robots/${targetId}/command`,
      { target, action },
      { headers: authHeaders }
    );
    return response.data;
  } catch (error) {
    console.log('전송 시도 URL:', `${API_BASE_URL}/robots/${targetId}/command`);
    console.log('토큰 로드 여부:', API_TOKEN ? '성공' : '실패(undefined)');
    throw error;
  }
};

export const verifyQrToken = async (token: string): Promise<AuthVerifyResult> => {
  const response = await axios.get(`${API_BASE_URL}/auth/verify`, {
    headers: createTokenHeaders(token),
  });
  return response.data;
};

export const fetchRobotMap = async (robotId?: string): Promise<RobotMap> => {
  const targetId = getRobotId(robotId);
  const response = await axios.get(`${API_BASE_URL}/robots/${targetId}/map`, {
    headers: authHeaders,
  });
  return response.data;
};

export const fetchRobotZones = async (robotId?: string): Promise<RobotZone[]> => {
  const targetId = getRobotId(robotId);
  const response = await axios.get(`${API_BASE_URL}/robots/${targetId}/zones`, {
    headers: authHeaders,
  });
  return response.data.zones || [];
};

export const saveRobotZones = async (robotId: string | undefined, zones: RobotZone[]) => {
  const targetId = getRobotId(robotId);
  const response = await axios.put(
    `${API_BASE_URL}/robots/${targetId}/zones`,
    { zones },
    { headers: authHeaders }
  );
  return response.data;
};

export const fetchZoneAirQuality = async (robotId?: string): Promise<ZoneAirQuality[]> => {
  const targetId = getRobotId(robotId);
  const response = await axios.get(`${API_BASE_URL}/robots/${targetId}/air-quality/zones`, {
    headers: authHeaders,
  });
  return response.data.zones || [];
};

export const fetchRobotStatus = async (robotId?: string): Promise<RobotStatusSummary> => {
  const targetId = getRobotId(robotId);
  const response = await axios.get(`${API_BASE_URL}/robots/${targetId}/status`, {
    headers: authHeaders,
  });
  return response.data;
};

export const saveRobotSchedule = async (
  robotId: string | undefined,
  payload: RobotSchedulePayload
) => {
  const targetId = getRobotId(robotId);
  const response = await axios.post(`${API_BASE_URL}/robots/${targetId}/schedule`, payload, {
    headers: authHeaders,
  });
  return response.data;
};

export const resetRobotData = async (robotId: string | undefined, target: 'MAP' | 'AI') => {
  const targetId = getRobotId(robotId);
  const response = await axios.post(
    `${API_BASE_URL}/robots/${targetId}/reset`,
    { target },
    { headers: authHeaders }
  );
  return response.data;
};

export const navigateRobot = async (
  robotId: string | undefined,
  payload: { type: 'COORDINATE'; x: number; y: number } | { type: 'ZONE'; zone_id: number }
) => {
  const targetId = getRobotId(robotId);
  const response = await axios.post(`${API_BASE_URL}/robots/${targetId}/navigate`, payload, {
    headers: authHeaders,
  });
  return response.data;
};

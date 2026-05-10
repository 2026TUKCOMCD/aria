import axios from 'axios';

export const sendRobotCommand = async (robotId: string, target: string, action: string) => {
  // 접두사를 붙인 이름으로 호출
  const API_BASE_URL = import.meta.env.VITE_ARIA_API_URL;
  const API_TOKEN = import.meta.env.VITE_API_SECRET_TOKEN;
  const DEFAULT_ID = import.meta.env.VITE_ROBOT_ID;

  const targetId = robotId || DEFAULT_ID || "1";

  try {
    const response = await axios.post(
      `${API_BASE_URL}/robots/${targetId}/command`,
      { target, action },
      {
        headers: {
          'Content-Type': 'application/json',
          'X-ARIA-SECRET': API_TOKEN, // 이제 undefined가 아니라 실제 값이 들어갑니다!
        },
      }
    );
    return response.data;
  } catch (error) {
    // 401 에러가 또 나면 아래 로그로 값이 찍히는지 확인해보세요
    console.log("전송 시도 URL:", `${API_BASE_URL}/robots/${targetId}/command`);
    console.log("토큰 로드 여부:", API_TOKEN ? "성공" : "실패(undefined)");
    throw error;
  }
};
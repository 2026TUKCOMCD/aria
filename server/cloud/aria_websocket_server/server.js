//  1. 필요한 모듈 불러오기 (Python의 import와 같음)
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

//  2. 서버 뼈대 만들기
const app = express();
const server = http.createServer(app);

//  3. WebSocket(확성기) 설정 (CORS 허용: 누구나 접속 가능하게 끔)
const io = new Server(server, {
    cors: { origin: "*" }
});

// JSON 데이터를 읽을 수 있게 설정 (API Gateway 설정과 비슷함)
app.use(express.json());

// ==========================================
// 파트 A: 클라이언트(웹앱)와 연결되는 부분
// ==========================================
io.on('connection', (socket) => {
    // 누군가 웹앱을 켜서 연결되면 이 로그가 찍힙니다.
    console.log('새로운 기기가 연결되었습니다! ID:', socket.id);

    // 웹앱을 끄거나 연결이 끊기면 찍히는 로그
    socket.on('disconnect', () => {
        console.log('기기 연결이 끊어졌습니다.');
    });
});

// ==========================================
// 파트 B: Lambda가 알림을 보낼 때 받는 API 엔드포인트
// ==========================================
// Lambda가 POST 방식으로 /api/alert 주소로 데이터를 보내면 여기가 실행됩니다.
app.post('/api/alert', (req, res) => {
    const alertData = req.body; // Lambda가 보낸 데이터 (예: { message: "먼지 나쁨" })
    console.log('Lambda에서 알림 도착:', alertData);

    // 핵심 포인트: io.emit()
    // 현재 접속해 있는 "모든" 웹앱에게 'robot_alert'라는 이름으로 데이터를 확성기로 쏴줍니다!
    io.emit('robot_alert', alertData);

    // Lambda에게 200 OK 응답을 돌려줍니다.
    res.status(200).json({ success: true, message: '클라이언트들에게 알림 전송 완료!' });
});

// ==========================================
//  서버 켜기 (포트 3000번)
// ==========================================
server.listen(3000, () => {
    console.log('ARIA WebSocket 서버가 3000번 포트에서 실행 중입니다!');
});
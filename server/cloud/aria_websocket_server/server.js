//  1. 필요한 모듈 불러오기 (Python의 import와 같음)
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
// PostgreSQL DB 통역기 불러오기
const { Pool } = require('pg') 

//=====================================
// DB 연결 셋업 (프라이빗 서브넷의 DB 정보 삽입)
// ====================================
const pool = new Pool({
    host: '10.0.1.183',
    user: 'aria_web',
    password: 'raspberryraspberry',
    database: 'postgres',
    port: 5432
})

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
app.post('/api/alert', async(req, res) => {
    const alertData = req.body; // Lambda가 보낸 데이터 (예: { message: "먼지 나쁨" })
    console.log('Lambda에서 알림 도착:', alertData);

    try{
        //DB에 저장 로직
        const query = `
      INSERT INTO robot_event_logs (robot_id, event_type, message, created_at)
      VALUES ($1, $2, $3, NOW())
    `;
    // 람다가 보내는 데이터 구조에 맞게 매핑 (없으면 기본값 처리)
    const values = [
      alertData.robot_id || 'aria-01', 
      alertData.event_type || 'INFO', 
      alertData.message || JSON.stringify(alertData)
    ];
    
    await pool.query(query, values);
    console.log('DB 저장 완료!');
    // 현재 접속해 있는 "모든" 웹앱에게 'robot_alert'라는 이름으로 데이터를 확성기로 쏴줍니다!
    io.emit('robot_alert', alertData);

    // Lambda에게 200 OK 응답을 돌려줍니다.
    res.status(200).json({ success: true, message: '클라이언트들에게 알림 전송 완료!' });
    } catch (error) {
        console.error('DB 저장 중 에러 발생:', error);
        // DB 저장에 실패하더라도 람다가 재시도하지 않도록 일단 500 에러를 반환합니다.
        res.status(500).json({ success: false, error: 'DB 저장 실패' });
    }
    
});

// ==========================================
// 파트 C: 프론트엔드 초기 화면용 과거 로그 조회 API (GET)
// ==========================================
// 프론트엔드가 GET 방식으로 /api/events 주소를 찌르면 실행됩니다.
app.get('/api/events', async (req, res) => {
    try {
        console.log('프론트엔드에서 과거 로그 조회 요청 도착!');
        
        // 프론트에서 특정 로봇 ID를 요구할 경우를 대비 (기본값: aria-01)
        const robotId = req.query.robot_id || 'aria_robot01';

        // 타임라인 방식: 해당 로봇의 전체 로그 중 최신 7개 가져오기
        const query = `
            SELECT log_id, event_type, message, created_at 
            FROM robot_event_logs 
            WHERE robot_id = $1 
            ORDER BY created_at DESC 
            LIMIT 7
        `;

        // DB에 쿼리 날리기 ($1 자리에 robotId가 쏙 들어갑니다)
        const { rows } = await pool.query(query, [robotId]);

        // 프론트엔드에게 성공 메시지와 함께 데이터를 JSON으로 던져줌
        res.status(200).json({
            success: true,
            data: rows
        });

    } catch (error) {
        console.error('로그 조회 중 에러 발생:', error);
        res.status(500).json({ success: false, error: 'DB 조회 실패' });
    }
});

// ==========================================
//  서버 켜기 (포트 3000번)
// ==========================================
server.listen(3000, () => {
    console.log('ARIA WebSocket 서버가 3000번 포트에서 실행 중입니다!');
});
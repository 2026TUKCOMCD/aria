//  1. 필요한 모듈 불러오기 (Python의 import와 같음)
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
// PostgreSQL DB 통역기 불러오기
const { Pool } = require('pg') 
const cors = require('cors'); // [추가] 브라우저 차단(CORS) 해결을 위한 모듈

//=====================================
// DB 연결 셋업 (프라이빗 서브넷의 DB 정보 삽입)
// ====================================
const pool = new Pool({
    host: '10.0.1.183',
    user: 'aria_web',
    password: 'raspberryraspberry',
    database: 'postgres',
    port: 5432,
    connectionTimeoutMillis: 2000, 
    idleTimeoutMillis: 1000
})

//  2. 서버 뼈대 만들기
const app = express();
app.use(cors());
app.use(express.json());
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
    const alertData = req.body;
    console.log('Lambda에서 알림 도착:', alertData);

    try {
        /* [임시 주석 처리] 로컬 테스트 중에는 DB 접속이 안 되므로 주석 처리합니다.
        const query = `
            INSERT INTO robot_event_logs (robot_id, event_type, message, created_at)
            VALUES ($1, $2, $3, NOW())
        `;
        const values = [
            alertData.robot_id || 'aria-01', 
            alertData.event_type || 'INFO', 
            alertData.message || JSON.stringify(alertData)
        ];
        await pool.query(query, values);
        console.log('DB 저장 완료!');
        */

        // [핵심] DB 저장이 실패하든 말든, 일단 웹으로 데이터를 쏩니다!
        console.log('웹으로 실시간 알림 전송 중...');
        io.emit('robot_alert', alertData); 

        res.status(200).json({ success: true, message: '클라이언트들에게 알림 전송 완료!' });
    } catch (error) {
        console.error('에러 발생:', error);
        // 에러가 나더라도 일단 웹에는 띄워보고 싶다면 위 io.emit을 catch 밖으로 빼도 됩니다.
        res.status(500).json({ success: false, error: '처리 실패' });
    }
});

// ==========================================
// 파트 C: 과거 로그 조회 API (GET) - DB 에러 방어 버전
// ==========================================
app.get('/api/events', async (req, res) => {
    const robotId = req.query.robot_id || 'aria_robot01';
    console.log(`프론트엔드에서 [${robotId}] 과거 로그 조회 요청 도착!`);

    try {
        // 1. DB 쿼리 시도
        const query = `
            SELECT log_id, event_type, message, created_at 
            FROM robot_event_logs 
            WHERE robot_id = $1 
            ORDER BY created_at DESC 
            LIMIT 7
        `;

        // DB 연결이 안 되는 환경(로컬)에서는 여기서 에러가 발생하여 catch문으로 넘어갑니다.
        const { rows } = await pool.query(query, [robotId]);

        res.status(200).json({
            success: true,
            data: rows
        });

    } catch (error) {
        // 2. DB 연결 실패 시 실행되는 구역 (로컬 테스트용)
        console.error('로그 조회 중 DB 연결 실패 (로컬 테스트 모드):', error.message);
        
        // 실제 DB 대신 프론트엔드에 전달할 가짜 데이터입니다.
        const dummyRows = [
            { log_id: 999, event_type: 'INFO', message: '현재 DB 연결이 불가능하여 테스트 데이터를 표시합니다.', created_at: new Date() },
            { log_id: 1, event_type: 'CLEANING', message: '요리 오염 감지 (과거 기록)', created_at: new Date(Date.now() - 3600000) },
            { log_id: 2, event_type: 'ACTIVITY', message: '활동 감지 (과거 기록)', created_at: new Date(Date.now() - 7200000) }
        ];

        // 상태 코드 200과 함께 가짜 데이터를 보내서 리액트가 멈추지 않게 합니다.
        res.status(200).json({
            success: true,
            data: dummyRows,
            isDummy: true
        });
    }
});

// ==========================================
//  서버 켜기 (포트 3000번)
// ==========================================
server.listen(3000, () => {
    console.log('ARIA WebSocket 서버가 3000번 포트에서 실행 중입니다!');
});
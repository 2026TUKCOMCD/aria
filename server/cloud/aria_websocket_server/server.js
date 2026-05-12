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

app.get('/auth/verify', (req, res) => {
    const authHeader = req.headers.authorization || '';
    const bearerToken = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
    const qrToken = req.headers['x-aria-qr-token'] || bearerToken;

    if (!qrToken || qrToken === 'invalid') {
        return res.status(200).json({
            valid: false,
            robot_id: null,
            user_name: null,
            robot_name: null
        });
    }

    res.status(200).json({
        valid: true,
        robot_id: '1',
        user_name: '민재',
        robot_name: 'ARIA_01'
    });
});

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
// ==========================================
// Part D: Map and zone APIs for the web app
// ==========================================
// Temporary in-memory data. Replace this with DB/S3 queries when the backend is ready.
const zoneStore = {
    '1': [
        {
            id: 1,
            name: '거실',
            center: { x: 1.5, y: 1.2 },
            area: { x_min: -1.5, y_min: -0.6, x_max: 4.3, y_max: 3.0 }
        },
        {
            id: 2,
            name: '주방',
            center: { x: 6.5, y: 1.4 },
            area: { x_min: 4.5, y_min: -0.4, x_max: 8.7, y_max: 3.1 }
        },
        {
            id: 3,
            name: '안방',
            center: { x: 2.4, y: 5.6 },
            area: { x_min: -1.2, y_min: 3.4, x_max: 5.4, y_max: 7.6 }
        }
    ]
};

const scheduleStore = {};

app.get('/mock-map.svg', (req, res) => {
    res.type('image/svg+xml').send(`
        <svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
            <rect width="800" height="600" fill="#f8fafc"/>
            <rect x="40" y="40" width="720" height="520" rx="12" fill="#ffffff" stroke="#1f2937" stroke-width="18"/>
            <path d="M400 40v520M40 310h720" stroke="#1f2937" stroke-width="12"/>
            <path d="M400 310h120M400 190h90M270 310v120" stroke="#f8fafc" stroke-width="18"/>
            <rect x="74" y="75" width="292" height="200" rx="8" fill="#e0f2fe" stroke="#60a5fa" stroke-width="4"/>
            <rect x="438" y="75" width="286" height="200" rx="8" fill="#dcfce7" stroke="#22c55e" stroke-width="4"/>
            <rect x="74" y="345" width="292" height="178" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="4"/>
            <rect x="438" y="345" width="286" height="178" rx="8" fill="#fee2e2" stroke="#ef4444" stroke-width="4"/>
            <text x="220" y="180" text-anchor="middle" font-size="36" font-weight="800" fill="#1f2937">거실</text>
            <text x="581" y="180" text-anchor="middle" font-size="36" font-weight="800" fill="#1f2937">주방</text>
            <text x="220" y="445" text-anchor="middle" font-size="36" font-weight="800" fill="#1f2937">안방</text>
            <text x="581" y="445" text-anchor="middle" font-size="36" font-weight="800" fill="#1f2937">작업방</text>
        </svg>
    `);
});

app.get('/robots/:id/map', (req, res) => {
    const robotId = req.params.id;

    res.status(200).json({
        robot_id: robotId,
        map_name: 'local test map',
        map_url: 'http://localhost:3000/mock-map.svg',
        metadata: {
            resolution: 0.025,
            origin: [-2.0, -1.0, 0.0],
            width: 800,
            height: 600
        },
        last_updated: new Date().toISOString()
    });
});

app.get('/robots/:id/zones', (req, res) => {
    const robotId = req.params.id;
    const zones = zoneStore[robotId] || [];

    res.status(200).json({
        robot_id: robotId,
        zones
    });
});

app.get('/robots/:id/status', (req, res) => {
    const robotId = req.params.id;

    res.status(200).json({
        robot_status: {
            battery: 82,
            is_charging: false,
            power: 'OFF',
            mode: 'AUTO',
            current_zone: 'LIVING_ROOM'
        },
        air_quality: {
            score: 75,
            grade: 'NORMAL',
            sensors: {
                pm25: 25.4,
                voc: 120,
                temperature: 24.5,
                humidity: 45.0
            }
        },
        robot_id: robotId
    });
});

app.post('/robots/:id/schedule', (req, res) => {
    const robotId = req.params.id;
    const { wake_time, sleep_time, enabled } = req.body;
    const timePattern = /^([01]\d|2[0-3]):[0-5]\d$/;

    if (!timePattern.test(wake_time) || !timePattern.test(sleep_time)) {
        return res.status(400).json({
            success: false,
            error: 'wake_time and sleep_time must be HH:mm format'
        });
    }

    scheduleStore[robotId] = {
        wake_time,
        sleep_time,
        enabled: Boolean(enabled),
        updated_at: new Date().toISOString()
    };

    console.log(`[${robotId}] 스케줄 저장 완료:`, scheduleStore[robotId]);

    res.status(200).json({
        success: true,
        message: '스케줄이 저장되었습니다.',
        data: scheduleStore[robotId]
    });
});

app.post('/robots/:id/reset', (req, res) => {
    const robotId = req.params.id;
    const { target } = req.body;

    if (target !== 'MAP' && target !== 'AI') {
        return res.status(400).json({
            success: false,
            error: 'target must be MAP or AI'
        });
    }

    if (target === 'MAP') {
        zoneStore[robotId] = [];
    }

    console.log(`[${robotId}] ${target} 초기화 명령 접수`);

    res.status(200).json({
        success: true,
        message: target === 'MAP'
            ? '맵 초기화 명령이 접수되었습니다.'
            : 'AI 초기화 명령이 접수되었습니다.'
    });
});

app.put('/robots/:id/zones', (req, res) => {
    const robotId = req.params.id;
    const zones = req.body.zones;

    if (!Array.isArray(zones)) {
        return res.status(400).json({ success: false, error: 'zones must be an array' });
    }

    zoneStore[robotId] = zones;
    console.log(`[${robotId}] zone 저장 완료:`, zones);

    res.status(200).json({ success: true });
});

app.get('/robots/:id/air-quality/zones', (req, res) => {
    const robotId = req.params.id;
    const now = Date.now();
    const zones = zoneStore[robotId] || [];
    const statuses = ['GOOD', 'NORMAL', 'BAD'];

    res.status(200).json({
        robot_id: robotId,
        update_interval_sec: 30,
        zones: zones.map((zone, index) => ({
            zone_id: zone.id,
            pm25: index === 0 ? 8 : index === 1 ? 22 : 48,
            voc: index === 0 ? 120 : index === 1 ? 310 : 720,
            status: statuses[index % statuses.length],
            measured_at: new Date(now - index * 90000).toISOString()
        }))
    });
});

app.post('/robots/:id/navigate', (req, res) => {
    const robotId = req.params.id;
    const command = req.body;

    if (!command || !command.type) {
        return res.status(400).json({ success: false, error: 'navigation type is required' });
    }

    console.log(`[${robotId}] 이동 명령 접수:`, command);
    io.emit('status_change', 'RUNNING');
    io.emit('robot_alert', {
        robot_id: robotId,
        event_type: 'NAVIGATE',
        message: command.type === 'ZONE'
            ? `구역 ${command.zone_id}으로 이동 명령을 접수했습니다.`
            : `좌표 (${command.x}, ${command.y})로 이동 명령을 접수했습니다.`
    });

    res.status(202).json({ success: true, accepted: true });
});

server.listen(3000, () => {
    console.log('ARIA WebSocket 서버가 3000번 포트에서 실행 중입니다!');
});

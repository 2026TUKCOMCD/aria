const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { Pool } = require('pg');
const cors = require('cors');

const pool = new Pool({
    host: '10.0.1.183',
    user: 'aria_web',
    password: 'raspberryraspberry',
    database: 'postgres',
    port: 5432,
    connectionTimeoutMillis: 2000,
    idleTimeoutMillis: 1000
});

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: '*' }
});

const sseClients = new Map();

const sendSseEvent = (res, eventName, data) => {
    res.write(`event: ${eventName}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
};

const broadcastSseEvent = (robotId, eventName, data) => {
    const targets = sseClients.get(robotId) || new Set();
    const broadcastTargets = sseClients.get('*') || new Set();

    [...targets, ...broadcastTargets].forEach((res) => {
        sendSseEvent(res, eventName, data);
    });
};

io.on('connection', (socket) => {
    console.log('웹 클라이언트가 연결되었습니다. ID:', socket.id);

    socket.on('disconnect', () => {
        console.log('웹 클라이언트 연결이 끊어졌습니다.');
    });
});

app.get('/health', (req, res) => {
    res.status(200).json({ ok: true, service: 'aria-websocket-server' });
});

app.get('/robots/:id/events/stream', (req, res) => {
    const robotId = req.params.id || '*';

    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    });

    sendSseEvent(res, 'connected', {
        type: 'CONNECTED',
        timestamp: new Date().toISOString(),
        message: '이벤트 스트림에 연결되었습니다.'
    });

    if (!sseClients.has(robotId)) {
        sseClients.set(robotId, new Set());
    }
    sseClients.get(robotId).add(res);
    console.log(`[${robotId}] SSE 클라이언트가 연결되었습니다.`);

    const keepAlive = setInterval(() => {
        sendSseEvent(res, 'ping', {
            type: 'PING',
            timestamp: new Date().toISOString()
        });
    }, 30000);

    req.on('close', () => {
        clearInterval(keepAlive);
        sseClients.get(robotId)?.delete(res);
        console.log(`[${robotId}] SSE 클라이언트 연결이 종료되었습니다.`);
    });
});

app.post('/api/alert', async (req, res) => {
    const alertData = req.body;
    console.log('Lambda 알림 수신:', alertData);

    try {
        io.emit('robot_alert', alertData);
        broadcastSseEvent(String(alertData.robot_id || '*'), alertData.event || 'clean_status', {
            type: alertData.type || alertData.event_type || 'INFO',
            timestamp: alertData.timestamp || new Date().toISOString(),
            message: alertData.message || '이벤트가 발생했습니다.',
            ...alertData
        });

        res.status(200).json({
            success: true,
            message: '알림을 웹 클라이언트로 전송했습니다.'
        });
    } catch (error) {
        console.error('알림 중계 실패:', error);
        res.status(500).json({
            success: false,
            error: '알림 중계 실패'
        });
    }
});

app.get('/api/events', async (req, res) => {
    const robotId = req.query.robot_id || 'aria_robot01';
    console.log(`[${robotId}] 이벤트 로그 조회 요청`);

    try {
        const query = `
            SELECT log_id, event_type, message, created_at
            FROM robot_event_logs
            WHERE robot_id = $1
            ORDER BY created_at DESC
            LIMIT 7
        `;
        const { rows } = await pool.query(query, [robotId]);

        res.status(200).json({
            success: true,
            data: rows
        });
    } catch (error) {
        console.error('이벤트 로그 조회 실패:', error.message);

        res.status(200).json({
            success: true,
            data: [
                {
                    log_id: 999,
                    event_type: 'INFO',
                    message: '이벤트 로그 DB 연결이 불가능하여 테스트 데이터를 표시합니다.',
                    created_at: new Date()
                }
            ],
            isDummy: true
        });
    }
});

server.listen(3000, () => {
    console.log('ARIA WebSocket 서버가 3000번 포트에서 실행 중입니다.');
});

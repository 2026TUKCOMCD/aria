#!/bin/bash

# 파이썬 실행기 및 스크립트 절대 경로 설정
PYTHON_EXEC="/usr/bin/python3"
SCRIPT_PATH="$HOME/aria_ai_system/scripts/night_training.py"
LOG_PATH="$HOME/aria_ai_system/data/night_training.log"

# 크론 표현식: 매일 새벽 3시 0분에 실행 (0 3 * * *)
# 실행 결과(print 출력문 등)는 night_training.log 파일에 누적 저장됩니다.
CRON_JOB="0 3 * * * $PYTHON_EXEC $SCRIPT_PATH >> $LOG_PATH 2>&1"

# 기존 크론탭 목록을 가져오되, 현재 스크립트가 이미 등록되어 있다면 제외(중복 방지)하고 새롭게 추가합니다.
(crontab -l 2>/dev/null | grep -v -F "$SCRIPT_PATH"; echo "$CRON_JOB") | crontab -

echo "===================================================="
echo "✅ 심야 자동 학습 스케줄러(Crontab) 등록이 완료되었습니다!"
echo "⏰ 매일 새벽 3시에 백그라운드에서 AI 모델이 자동으로 재학습됩니다."
echo "📝 실행 로그 확인 명령어: tail -f $LOG_PATH"
echo "===================================================="

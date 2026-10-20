#!/bin/bash
# ARIA 심야 자동 학습 등록 스크립트
#
# - 이 스크립트가 있는 폴더(저장소 안)의 night_training.py를 cron이 직접 실행한다.
#   → git pull만 하면 다음 날 새벽부터 새 학습 코드가 쓰인다(따로 복사할 필요 없음).
# - 예전에 등록된 night_training.py 항목(복사본 경로 포함)은 모두 지우고 새로 등록한다.
# - 등록 직후 한 번 바로 학습해서, 새 형식 모델을 지금 만들어 둔다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/usr/bin/python3"
SCRIPT_PATH="$SCRIPT_DIR/night_training.py"
LOG_DIR="$HOME/aria_ai_system/data"
LOG_PATH="$LOG_DIR/night_training.log"
mkdir -p "$LOG_DIR"

# 매일 새벽 3시 0분 실행
CRON_JOB="0 3 * * * cd $SCRIPT_DIR && $PYTHON_EXEC $SCRIPT_PATH >> $LOG_PATH 2>&1"

# 기존 night_training.py 항목(어느 경로든) 제거 후 새로 추가
(crontab -l 2>/dev/null | grep -v "night_training.py"; echo "$CRON_JOB") | crontab -

echo "===================================================="
echo "✅ 심야 자동 학습 등록 완료 (매일 03:00)"
echo "   실행 파일: $SCRIPT_PATH"
echo "   로그:      tail -f $LOG_PATH"
echo "===================================================="
echo "🧠 지금 바로 한 번 학습합니다..."
cd "$SCRIPT_DIR" && "$PYTHON_EXEC" "$SCRIPT_PATH"

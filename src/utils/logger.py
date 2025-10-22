# src/utils/logger.py

import csv
import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "conversation_log.csv")

def log_to_csv(user_input, ai_output, agent_used):
    """
    사용자의 질문, AI의 답변, 사용된 Agent를 CSV 파일에 기록합니다.
    """
    # 1. 'logs' 폴더가 없으면 생성합니다.
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 2. 파일이 존재하는지 확인합니다.
    file_exists = os.path.isfile(LOG_FILE)
    
    try:
        # 3. 파일을 추가 모드('a')로 열고, CSV writer를 설정합니다.
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 4. 파일이 새로 생성되었다면, 헤더(제목 줄)를 먼저 씁니다.
            if not file_exists:
                writer.writerow(["Timestamp", "UserInput", "AI_Output", "AgentUsed"])
            
            # 5. 현재 시간, 사용자 입력, AI 답변, 사용된 Agent를 한 줄로 기록합니다.
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, user_input, ai_output, agent_used])
            
    except Exception as e:
        print(f"--- 🚨 CSV 로깅 에러 발생: {e} ---")
import logging
logger = logging.getLogger('logger') # 적지 않으면 root로 생성

# 2. logging level 지정 - 기본 level Warning
logger.setLevel(logging.INFO)

# 3. logging formatting 설정 - 문자열 format과 유사 - 시간, logging 이름, level - messages
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] >> %(message)s')

# 4. handler : log message를 지정된 대상으로 전달하는 역할.
# SteamHandler : steam(terminal 같은 console 창)에 log message를 보냄
# FileHandler : 특정 file에 log message를 보내 저장시킴.
# handler 정의
stream_handler = logging.StreamHandler()
# handler에 format 지정
steam_handler.setFormatter(formatter)
# logger instance에 handler 삽입
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('log.txt', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 기록하면된다.

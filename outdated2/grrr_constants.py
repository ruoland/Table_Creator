# 테이블 구조 관련 상수
MIN_COLS, MAX_COLS = 2, 12  # 열의 최소 및 최대 개수 (기존: 3, 10)
MIN_ROWS, MAX_ROWS = 2, 15  # 행의 최소 및 최대 개수 (기존: 3, 10)
BASE_CELL_WIDTH, BASE_CELL_HEIGHT = 60, 30  # 기본 셀 크기 (기존: 40, 20)
MIN_CELL_SIZE = 40  # 최소 셀 크기 (변경 없음)
MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT = 400, 600  # 최소 이미지 크기 (기존: 300, 500)
MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT = 2048, 2048  # 최대 이미지 크기 (기존: 1200, 1200)

# 배치 및 처리 관련 상수
BATCH_SIZE = 1000  # 배치 크기 (변경 없음)
NUM_WORKERS = 4  # 병렬 처리를 위한 워커 수 (새로 추가)
OUTER_LINE_PROBABILITY = 0.8  # 80%의 확률로 외곽선을 그립니다.
ROTATION_PROBABILITY = 0.2
MAX_ROTATION_ANGLE = 5
NOISE_PROBABILITY = 0.3
CONTRAST_ADJUSTMENT_PROBABILITY = 0.2
# 스타일 관련 상수
STYLES = ['thin', 'medium', 'thick', 'double']  # 선 스타일 (변경 없음)
FONTS = ['fonts/NanumGothic.ttf', 'fonts/SANGJU Dajungdagam.ttf', 'fonts/SOYO Maple Regular.ttf']  # 폰트 (변경 없음)
LINE_STYLES = ['solid', 'dashed', 'dotted']  # 선 스타일 (변경 없음)
BACKGROUND_COLORS = {
    'light': {'white': 255, 'light_gray': 250, 'beige': 245, 'light_yellow': 253},  # 밝은 배경색 추가
    'dark': {'black': 0, 'dark_gray': 25, 'navy': 15, 'dark_green': 20}  # 어두운 배경색 추가
}

# 콘텐츠 관련 상수
DAYS = ['월요일', '화요일', '수요일', '목요일', '금요일', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', '월', '화', '수', '목', '금', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
TIMES = ['1교시', '2교시', '3교시', '4교시', '5교시', '6교시', '7교시', '1st Period', '2nd Period', '3rd Period', '4th Period', '5th Period', '6th Period', '7th Period']
GRADES = ['1학년', '2학년', '3학년', '4학년', '1st Grade', '2nd Grade', '3rd Grade', '4th Grade']
COMMON_WORDS = ['Hello', 'Table', 'Data', 'Analysis', 'Information',
    'Computer', 'AI', 'Machine', 'Deep Learning', 'Big Data',
    'Progrng', 'Algorithm', 'Network', 'Database', 'Security',
    
    # 숫자
    '123', '456', '789', '1000', '10000',
    '1.23', '4.56', '7.89', '0.01', '99.99',
    
    # 특수문자
    '!@#$%', '&*()_+', '[]{};:,.<>?', '~`-=', '|\\',

    # 혼합
    'A1B2C3', '가나다123', 'ABC가나다', '123ABC가나다',

    # 여러 줄 (줄바꿈 문자 포함)
    'First line\nSecond line', '첫째 줄\n둘째 줄',
    'Line 1\nLine 2\nLine 3', '줄 1\n줄 2\n줄 3',
    
    # 긴 단어
    'Supercalis',
    '청춘예찬',
    # 짧은 단어
    'a', 'b', 'c', '가', '나', '다',

    # 공백 포함
    'Hello World', '안녕 세상', 'Open AI GPT', '인공 지능 모델',
    # 수식 (LaTeX 스타일)
    'E = mc^2', 'f(x) = ax^2 + bx + c', '\\sum_{i=1}^n i = \\frac{n(n+1)}{2}',
    
    # URL 및 이메일
    'https://ww', 'user@exom',
    
    # 날짜 및 시간
    '2023-09-13', '14:30:00', '230분',
    
    # 통화
    '₩10,000', '$100.00', '€50.00', '¥1000',
    '수학', '영어', '국어', '과학', '사회', '체육', '음악', '미술',
    '월', '화', '수', '목', '금', '토', '일',
    '1교시', '2교시', '3교시', '4교시', '5교시', '6교시', '7교시',
    '점심', '휴식', '자습', '회의', '상담',
    '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일',
    '김', '이', '박', '최', '정', '강', '조', '윤', '장', '임',
    '교실', '강당', '도서관', '운동장', '식당',
    '시험', '과제', '발표', '토론', '실험', '프로젝트',
    '아침', '점심', '저녁', '오전', '오후',
    '학교', '학원', '집', '회사', '병원', '은행', '마트',
    '회의', '미팅', '약속', '일정', '계획', '목표',
    '집에', '가자', '데이터', '공학', '기계', '전기', '전자', '컴퓨터', '소프트웨어',
    '하드웨어', '설계', '실험', '프로젝트', '프로그래밍', '알고리즘', '데이터', '로봇',
    '미술', '디자인', '드로잉', '페인팅', '조각', '일러스트레이션', '색채',
    '구도', '작품', '전시회', '포트폴리오',
    '농업', '식물', '작물', '토양', '비료', '농기계', '재배', '수확',
    '농촌', '환경', '생태계',
    '체육', '운동', '스포츠', '훈련', '경기', '체력', '건강', '스트레칭',
    '팀워크', '대회', '선수', '코치',
]
# 학과명
DEPARTMENTS = ['컴퓨터공학과', '전자공학과', '기계공학과', '화학공학과', '생명공학과', '경영학과', '경제학과', '심리학과', 
               '사회학과', '철학과', '역사학과', '영문학과', '국문학과', '물리학과', '수학과', '통계학과', '의학과', 
               '간호학과', '약학과', '건축학과', '디자인학과', '음악학과', '체육학과', '교육학과', '법학과']

# 과목명
SUBJECTS = ['프로그래밍 기초', '데이터베이스 설계', '운영체제론', '컴퓨터네트워크', '인공지능 개론', '웹프로그래밍', '알고리즘 분석', 
            '데이터구조', '소프트웨어공학', '정보보안', '클라우드컴퓨팅', '빅데이터 분석', '머신러닝', '딥러닝', '사물인터넷',
            '양자컴퓨팅 입문', '블록체인 기술', 'augmented reality', 'virtual reality', '로보틱스', '컴퓨터비전',
            '자연어처리', '병렬컴퓨팅', '임베디드시스템', '컴퓨터그래픽스', '게임프로그래밍', '모바일앱개발', '사이버보안',
            '데이터마이닝', '컴퓨터구조', '소프트웨어테스팅', '프로젝트관리', '인간컴퓨터상호작용', '분산시스템']

# 교수명 (다양한 성씨 포함)
PROFESSORS = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임', '한', '오', '서', '신', '권', '황', '안', '송', '전', '홍', '유', '고', '문', '양', '손', '배', '조', '백', '허', '유', '남', '심', '노', '정', '하', '곽', '성', '차', '주', '우', '구', '신', '임', '전', '민', '유', '류', '나', '진', '지', '엄', '채', '원', '천', '방', '공', '강', '현', '함', '변', '염', '양', '변', '여', '추', '노', '도', '소', '신', '석', '선', '설', '마', '길', '주', '국', '여', '탁', '진', '승', '모', '도', '곽', '범', '어', '양', '옥', '국', '맹', '석', '은', '편', '풍', '금', '왕', '제']

# 건물명
BUILDINGS = ['공학관', '과학관', '인문관', '경영관', '예술관', '체육관', '도서관', '학생회관', '연구동', '강당', '기숙사', 
             '의학관', '법학관', '음악관', '건축관', 'IT융합센터', '창업보육센터', '국제교류관', '산학협력관', '미디어센터', 
             '박물관', '천문대', '수의과대학', '농업생명과학관', '생명공학관', '해양과학관', '환경생태관', '융합과학기술원']

# 강의실 유형
ROOM_TYPES = ['강의실', '실험실', '세미나실', '스튜디오', '컴퓨터실', '강당', '회의실', '연구실', '실습실', '자습실', '워크샵룸']

# 시간
TIMES = ['09:00-10:30', '10:30-12:00', '13:00-14:30', '14:30-16:00', '16:00-17:30', '17:30-19:00', 
         '19:00-20:30', '20:30-22:00', '08:00-09:30', '12:00-13:30', '15:00-16:30', '18:00-19:30']

# 수업 유형
CLASS_TYPES = ['이론', '실습', '세미나', '특강', '보강', '개론', '심화', '프로젝트', '워크샵', '연구', '캡스톤디자인', 
               '현장실습', '인턴십', '논문연구', '학술대회', '콜로키움', '학년프로젝트', '산학협력', '창업실습', '글로벌챌린지']

# 학사 관련 용어
ACADEMIC_TERMS = ['수강신청', '학점', '장학금', '졸업요건', '복수전공', '부전공', '연계전공', '학석사연계과정', '계절학기', 
                  '학점교류', '인턴십', '해외교환학생', '논문심사', '자격시험', '현장실습', '학회', '동아리', '봉사활동']

# 대학 행사
EVENTS = ['입학식', '졸업식', '축제', '학술제', '취업박람회', '오리엔테이션', '홈커밍데이', '체육대회', '학과설명회', 
          '연합동아리발표회', '캠퍼스투어', '외국인유학생환영회', '창업경진대회', '해커톤', '학생총회', '교수님과의대화', 
          '명사특강', '학부모초청행사', '연구성과발표회', '산학협력포럼']

# 생성 비율 관련 상수
MERGED_CELL_RATIO = 0.4  # 병합된 셀의 비율 (기존: 0.6)
SHAPE_GENERATION_RATIO = 0.15  # 도형 생성 비율 (기존: 0.2)
EMPTY_CELL_RATIO = 0.05  # 빈 셀의 비율 (기존: 0.1)
IMPERFECT_RATIO = 0.2  # 불완전한 이미지 비율 (기존: 0.1)
HEADER_ROW_RATIO = 0.8  # 헤더 행을 포함할 확률 (새로 추가)
HEADER_COL_RATIO = 0.6  # 헤더 열을 포함할 확률 (새로 추가)

# 불완전성 관련 상수
BLUR_PROBABILITY = 0.3  # 흐림 효과를 적용할 확률 (기존 코드의 값을 상수화)
MIN_BLUR, MAX_BLUR = 0.3, 0.7  # 흐림 효과의 최소 및 최대 강도 (기존 코드의 값을 상수화)
LINE_IMPERFECTION_PROBABILITY = 0.3  # 선 불완전성을 적용할 확률 (기존 코드의 값을 상수화)
MAX_IMPERFECT_LINES = 3  # 최대 불완전 선 개수 (기존 코드의 값을 상수화)
CELL_IMPERFECTION_PROBABILITY = 0.1  # 셀 불완전성을 적용할 확률 (기존 코드의 값을 상수화)

# 폰트 관련 상수
MIN_FONT_SIZE = 8  # 최소 폰트 크기 (기존 코드의 값을 상수화)
MAX_FONT_SIZE_RATIO = 0.25  # 최대 폰트 크기 비율 (셀 높이 대비) (기존 코드의 값을 상수화)
MIN_FONT_SIZE_RATIO = 0.08  # 최소 폰트 크기 비율 (셀 높이 대비) (기존 코드의 값을 상수화)

# 셀 내용 관련 상수
MIN_CELL_SIZE_FOR_TEXT = 30  # 텍스트를 넣기 위한 최소 셀 크기 (기존 코드의 값을 상수화)
TEXT_PADDING = 10  # 텍스트 주변 여백 (기존 코드의 값을 상수화)

# 노이즈 관련 상수
MIN_NOISE, MAX_NOISE = 5, 20  # 노이즈 개수 범위 (기존 코드의 값을 상수화)
MIN_NOISE_SIZE, MAX_NOISE_SIZE = 1, 3  # 노이즈 크기 범위 (기존 코드의 값을 상수화)
DOT_NOISE_PROBABILITY = 0.7  # 점 노이즈를 생성할 확률 (기존 코드의 값을 상수화)
MAX_LINE_NOISE_LENGTH = 5  # 선 노이즈의 최대 길이 (기존 코드의 값을 상수화)

# 도형 관련 상수
MIN_SHAPES_PER_CELL, MAX_SHAPES_PER_CELL = 1, 3  # 셀당 도형 개수 범위 (기존 코드의 값을 상수화)
SHAPE_TYPES = ['rectangle', 'ellipse', 'triangle', 'line', 'arc', 'polygon']  # 도형 종류 (기존 코드의 값을 상수화)

# 선 관련 상수
MIN_LINE_WIDTH, MAX_LINE_WIDTH = 1, 3  # 선 두께 범위 (새로 추가)

# 데이터 분할 비율
TRAIN_RATIO = 0.8  # 학습 데이터 비율 (변경 없음)
VAL_RATIO = 0.1  # 검증 데이터 비율 (변경 없음)
TEST_RATIO = 0.1  # 테스트 데이터 비율 (새로 추가)

# 랜덤 시드
RANDOM_SEED = 42  # 재현 가능성을 위한 랜덤 시드 (새로 추가)

# 이미지 저장 관련
IMAGE_FORMAT = 'PNG'  # 이미지 저장 형식 (새로 추가)
IMAGE_QUALITY = 95  # 이미지 품질 (JPEG의 경우) (새로 추가)

# 로깅 레벨
LOG_LEVEL = 'INFO'  # 로깅 레벨 설정 (새로 추가)
# 불완전성 관련 상수
BLUR_PROBABILITY = 0.3  # 흐림 효과를 적용할 확률
MIN_BLUR, MAX_BLUR = 0.3, 0.7  # 흐림 효과의 최소 및 최대 강도
LINE_IMPERFECTION_PROBABILITY = 0.3  # 선 불완전성을 적용할 확률
MAX_IMPERFECT_LINES = 3  # 최대 불완전 선 개수
CELL_IMPERFECTION_PROBABILITY = 0.1  # 셀 불완전성을 적용할 확률

# 노이즈 관련 상수
MIN_NOISE, MAX_NOISE = 5, 20  # 노이즈 개수 범위
MIN_NOISE_SIZE, MAX_NOISE_SIZE = 1, 3  # 노이즈 크기 범위
DOT_NOISE_PROBABILITY = 0.7  # 점 노이즈를 생성할 확률
MAX_LINE_NOISE_LENGTH = 5  # 선 노이즈의 최대 길이

# 도형 관련 상수
MIN_SHAPES_PER_CELL, MAX_SHAPES_PER_CELL = 1, 3  # 셀당 도형 개수 범위
SHAPE_TYPES = ['rectangle', 'ellipse', 'triangle', 'line', 'arc', 'polygon']  # 도형 종류

# 선 관련 상수
MIN_LINE_WIDTH, MAX_LINE_WIDTH = 1, 3  # 선 두께 범위

# 추가된 불완전성 관련 상수
PROTRUSION_PROBABILITY = 0.2  # 선에 돌기를 추가할 확률
DOT_PROBABILITY = 0.2  # 점 노이즈를 추가할 확률
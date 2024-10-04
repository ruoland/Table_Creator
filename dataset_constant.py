import  random

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

def generate_random_title():
    title_types = [
        # 기존 영어 타입
        "Table of {}", "Summary of {}", "{} Overview", "{} Statistics",
        "{} Data", "Annual {} Report", "Quarterly {} Results",
        "{} Analysis", "{} Comparison", "{} Trends",
        # 추가된 영어 타입
        "{} Metrics", "{} Forecast", "{} Distribution", "{} Breakdown",
        "Key {} Indicators", "{} Performance Review", "{} Benchmark",
        "{} Insights", "{} Evaluation", "{} Progress Report",
        # 한국어 타입
        "{}표", "{} 요약", "{} 개요", "{} 통계",
        "{} 데이터", "연간 {} 보고서", "분기별 {} 결과",
        "{} 분석", "{} 비교", "{} 동향",
        "{} 지표", "{} 예측", "{} 분포", "{} 내역",
        "주요 {} 지표", "{} 성과 검토", "{} 벤치마크",
        "{} 인사이트", "{} 평가", "{} 진행 보고서",
        # 대학 시간표 관련 타입
        "{} 강의 시간표", "{} 수업 일정", "{} 학기 일정표",
        "{} 강좌 스케줄", "{}과 커리큘럼", "{} 수강 계획표",
        
        # 회사 시간표 관련 타입
        "{} 근무 시간표", "{} 업무 일정", "{} 부서 스케줄",
        "{} 교대 근무표", "{} 회의 일정", "{} 프로젝트 타임라인",
        
        # 영어 버전
        "{} Class Schedule", "{} Course Timetable", "{} Semester Plan",
        "{} Work Shift Schedule", "{} Department Roster", "{} Meeting Planner",
        # 시간표 관련 타입
        "{} 시간표", "{} 스케줄", "{} 일정표", "{} 예약 현황",
        "{} 주간 일정", "{} 월간 계획", "{} 수업 시간",
        "{} 근무표", "{} 교대 일정", "{} 상담 예약",
        
        # 영어 버전
        "{} Timetable", "{} Schedule", "{} Planner", "{} Booking Status",
        "{} Weekly Plan", "{} Monthly Schedule", "{} Class Times",
        "{} Shift Roster", "{} Appointment Calendar"
    ]
    subjects = [
        # 기존 영어 주제
        "Sales", "Revenue", "Expenses", "Profit", "Growth", "Performance",
        "Inventory", "Customer", "Product", "Market", "Financial", "Operational",
        "Employee", "Project", "Research", "Development", "Quality", "Efficiency",
        # 추가된 영어 주제
        "Budget", "Investment", "Risk", "Compliance", "Innovation", "Sustainability",
        "Productivity", "Marketing", "Supply Chain", "HR", "IT", "Customer Satisfaction",
        "Cost Reduction", "Asset Management", "Strategic Planning", "Competitive Analysis",
        # 한국어 주제
        "매출", "수익", "비용", "이익", "성장", "실적",
        "재고", "고객", "제품", "시장", "재무", "운영",
        "직원", "프로젝트", "연구", "개발", "품질", "효율성",
        "예산", "투자", "리스크", "규정 준수", "혁신", "지속가능성",
        "생산성", "마케팅", "공급망", "인사", "정보기술", "고객 만족도",
        "비용 절감", "자산 관리", "전략 계획", "경쟁 분석",
        
         # 대학 관련 주제
        "1학년", "2학년", "3학년", "4학년", "학부", "대학원",
        "공과대학", "인문대학", "자연과학대학", "경영대학", "의과대학",
        "컴퓨터공학과", "경제학과", "심리학과", "생물학과", "물리학과",
        
        # 회사 관련 주제
        "영업부", "마케팅팀", "인사과", "개발팀", "고객서비스부", "경영지원팀",
        "주간", "야간", "A조", "B조", "C조", "임원", "관리자", "신입사원",
        
        # 영어 버전
        "Freshman", "Sophomore", "Junior", "Senior", "Undergraduate", "Graduate",
        "Engineering", "Humanities", "Natural Sciences", "Business", "Medicine",
        "Sales", "Marketing", "HR", "Development", "Customer Service", "Administration",
        "Day Shift", "Night Shift", "Team A", "Team B", "Team C", "Executives", "Managers", "New Hires",
        # 피트니스 및 운동 관련
        "필라테스", "요가", "헬스", "PT", "그룹 운동", "수영",
        "에어로빅", "스피닝", "크로스핏", "무술",
        
        # 교육 관련
        "정규 수업", "보충 수업", "특강", "세미나", "워크샵",
        "실험실 수업", "온라인 강의", "오프라인 강의",
        
        # 기업 및 사무직 관련
        "팀 미팅", "프로젝트 일정", "출장", "원격 근무",
        "연차 휴가", "교육 훈련", "성과 평가",
        
        # 서비스 산업 관련
        "레스토랑 예약", "헤어샵 예약", "병원 진료",
        "호텔 객실", "항공편", "공연",
        
        # 자유직 및 프리랜서 관련
        "프리랜서 업무", "클라이언트 미팅", "납품 일정",
        "작업 마감일", "인터뷰",
        
        # 영어 버전
        "Pilates", "Yoga", "Gym", "Personal Training", "Group Fitness",
        "Regular Classes", "Supplementary Classes", "Special Lectures",
        "Team Meetings", "Project Timeline", "Business Trips",
        "Restaurant Bookings", "Hair Salon Appointments", "Medical Consultations",
        "Freelance Work", "Client Meetings", "Delivery Schedule"
    ]
    return random.choice(title_types).format(random.choice(subjects))

COMMON_WORDS.extend([
    # 학교 생활 관련
    '등교', '하교', '수업', '쉬는 시간', '청소 시간', '조회', '종례', '방과 후', '동아리 활동',
    '학생회', '학부모회', '교직원 회의', '학년 총회', '반 친구', '담임 선생님', '교과 선생님',

    # 대학 생활 관련
    '신입생', '재학생', '졸업생', '휴학', '복학', '자퇴', '편입', '석사과정', '박사과정', '조교',
    '교수님 면담', '논문 지도', '학위 수여식', '연구실', '실험실', '도서관 열람실', '기숙사',

    # 직장 생활 관련
    '출근', '퇴근', '야근', '연차', '휴가', '출장', '재택근무', '유연근무제', '초과근무',
    '인사고과', '승진', '이직', '퇴직', '연봉 협상', '복리후생', '사내 교육', '워크숍',

    # 일정 관련
    '주간 회의', '월간 보고', '분기별 실적', '연간 계획', '프로젝트 킥오프', '중간 점검',
    '최종 발표', '데드라인', '마감 기한', '연기', '취소', '긴급 소집',

    # 건강 및 운동 관련
    '아침 운동', '저녁 운동', '스트레칭', '웨이트 트레이닝', '유산소 운동', '식단 관리',
    '체중 관리', '건강 검진', '예방 접종', '정기 검진', '재활 치료',

    # 식사 관련
    '아침 식사', '점심 식사', '저녁 식사', '간식', '야식', '다이어트 식단', '영양 식단',
    '채식', '육식', '패스트푸드', '배달 음식', '회식', '술자리',

    # 취미 및 여가 활동
    '독서', '영화 감상', '음악 감상', '그림 그리기', '요리', '베이킹', '원예', '등산',
    '낚시', '여행', '캠핑', '사진 촬영', '악기 연주', '댄스', '요가', '명상',

    # 기술 및 IT 관련
    'AI 학습', '코딩 연습', '앱 개발', '웹 디자인', '데이터 분석', '클라우드 컴퓨팅',
    '사이버 보안', '블록체인', 'VR/AR 개발', '로봇 프로그래밍', 'IoT 프로젝트',

    # 비즈니스 관련
    '시장 조사', '고객 미팅', '제품 출시', '마케팅 전략', '영업 목표', '재무 계획',
    '투자 유치', '주주 총회', 'IR', 'M&A', '기업 공개(IPO)',

    # 문화 및 엔터테인먼트
    '콘서트', '전시회', '뮤지컬', '연극', '영화 개봉', 'TV 프로그램', '라디오 쇼',
    '팬 미팅', '사인회', '북 콘서트', '토크 콘서트',

    # 사회 및 정치 관련
    '선거', '국회 회의', '정책 발표', '외교 회담', '정상 회담', '기자 회견',
    '여론 조사', '시민 단체 활동', '자원봉사', '기부',

    # 환경 및 지속가능성 관련
    '재활용', '친환경 활동', '탄소 중립', '에너지 절약', '생태계 보호', '기후 변화 대응',
    '지속가능한 발전', '환경 교육', '그린 캠페인',

    # 금융 및 경제 관련
    '주식 거래', '펀드 투자', '부동산 매매', '환율 변동', '금리 인상', '물가 상승',
    '경제 지표 발표', '세금 신고', '연말 정산', '예산 편성',

    # 기타
    '긴급 상황', '비상 대피', '재난 훈련', '안전 교육', '소방 훈련', '응급 처치',
    '교통 정보', '날씨 예보', '공휴일', '기념일', '축제', '행사',
])

# DEPARTMENTS, SUBJECTS, PROFESSORS, BUILDINGS, ROOM_TYPES, TIMES, CLASS_TYPES, ACADEMIC_TERMS, EVENTS에도 추가
DEPARTMENTS.extend(['인공지능학과', '로봇공학과', '환경공학과', '식품영양학과', '관광경영학과', '미디어커뮤니케이션학과'])
SUBJECTS.extend(['인공지능윤리', '지속가능발전과기술', '창의적문제해결', '글로벌이슈와리더십', '융합캡스톤디자인'])
PROFESSORS.extend(['남', '심', '노', '정', '하', '곽', '성', '차', '주', '우', '구'])
BUILDINGS.extend(['스마트팩토리', '창의융합센터', '글로벌라운지', '메이커스페이스', '혁신창업관'])
ROOM_TYPES.extend(['VR실습실', '3D프린팅실', '코워킹스페이스', '화상회의실', '멀티미디어실'])
TIMES.extend(['07:30-09:00', '21:00-22:30', '22:30-24:00'])
CLASS_TYPES.extend(['플립러닝', '팀티칭', '온오프혼합', 'MOOC', 'PBL(문제중심학습)'])
ACADEMIC_TERMS.extend(['학점은행제', '독립연구', '학생설계전공', '융합전공', '마이크로디그리'])
EVENTS.extend(['스타트업 데모데이', '글로벌 포럼', '산학협력 엑스포', '인공지능 경진대회', '친환경 캠퍼스 캠페인'])
COMMON_WORDS.extend([
    # 학교 생활 관련
    '등교시간에맞춰서일어나기', '하교후학원가기', '수업시간에집중하기', '쉬는시간에친구들과대화하기', '청소시간에교실정리하기', 
    '아침조회시간에출석체크하기', '종례시간에하루마무리하기', '방과후활동으로동아리참여하기', '동아리활동으로새로운취미배우기',
    '학생회활동으로리더십기르기', '학부모회모임에참석하기', '교직원회의에안건제출하기', '학년총회에서의견발표하기', 
    '반친구들과단체사진찍기', '담임선생님과상담시간갖기', '교과선생님께질문하기',

    # 대학 생활 관련
    '신입생오리엔테이션참석하기', '재학생등록금납부하기', '졸업생취업현황조사응답하기', '휴학신청서제출하기', '복학절차밟기', 
    '자퇴원서작성하기', '편입시험준비하기', '석사과정논문주제선정하기', '박사과정연구계획서작성하기', '조교로수업보조하기',
    '교수님면담시간예약하기', '논문지도받기', '학위수여식에가운입고참석하기', '연구실에서밤샘실험하기', '실험실안전교육이수하기', 
    '도서관열람실에서시험공부하기', '기숙사룸메이트와친해지기',

    # 직장 생활 관련
    '아침일찍출근하기', '정시에퇴근하기', '야근후저녁식사하기', '연차휴가신청서제출하기', '여름휴가계획세우기', '해외출장준비하기', 
    '재택근무시화상회의참여하기', '유연근무제활용하기', '초과근무수당정산하기', '인사고과평가받기', '승진인터뷰준비하기', 
    '이직을위한이력서업데이트하기', '퇴직금정산받기', '연봉협상을위한자료준비하기', '복리후생제도활용하기', '사내교육프로그램이수하기', 
    '팀워크숍에적극참여하기',

    # 일정 관련
    '주간회의안건준비하기', '월간보고서작성하기', '분기별실적발표준비하기', '연간계획수립하기', '프로젝트킥오프미팅주최하기', 
    '중간점검회의소집하기', '최종발표자료만들기', '데드라인맞추기위해야근하기', '마감기한연장요청하기', '일정연기에따른조정하기', 
    '갑작스런일정취소에대응하기', '긴급소집에빠르게대처하기',

    # 건강 및 운동 관련
    '아침운동으로조깅하기', '저녁운동으로수영가기', '스트레칭으로몸풀기', '웨이트트레이닝으로근력키우기', '유산소운동으로체지방감소하기', 
    '균형잡힌식단관리하기', '체중관리를위해칼로리계산하기', '정기건강검진예약하기', '독감예방접종맞기', '연례종합검진받기', 
    '물리치료로재활훈련받기',

    # 식사 관련
    '건강한아침식사준비하기', '점심식사로동료들과회식가기', '가족과함께저녁식사하기', '오후간식으로과일먹기', '야식은자제하기', 
    '다이어트식단으로샐러드먹기', '영양가높은식단구성하기', '채식위주의식단유지하기', '육식을줄이고생선먹기', 
    '패스트푸드대신집밥먹기', '배달음식주문하기', '회식자리에서술조절하기', '술자리에서안주선택하기',

    # 취미 및 여가 활동
    '취침전독서하기', '주말영화감상하기', '출퇴근길음악감상하기', '주말에그림그리기', '요리레시피연구하기', '베이킹으로빵만들기', 
    '베란다에서허브키우기', '주말등산으로체력기르기', '휴가때바다낚시가기', '해외여행계획세우기', '가족과함께캠핑가기', 
    '일상을사진으로담기', '악기연주로스트레스해소하기', '댄스클래스수강하기', '요가로몸과마음다스리기', '명상으로마음챙김하기',
    # 기술 및 IT 관련
    '인공지능알고리즘학습하기', '새로운프로그래밍언어배우기', '모바일앱개발프로젝트시작하기', '반응형웹디자인적용하기', 
    '빅데이터분석도구사용법익히기', '클라우드컴퓨팅서비스활용하기', '사이버보안대책마련하기', '블록체인기술이해하기', 
    '가상현실콘텐츠제작하기', '로봇프로그래밍기초배우기', '사물인터넷기기연결하기',

    # 비즈니스 관련
    '신규시장진출을위한조사하기', '잠재고객미팅준비하기', '신제품출시이벤트기획하기', '디지털마케팅전략수립하기', '연간영업목표설정하기', 
    '장기재무계획수립하기', '벤처캐피털투자유치준비하기', '정기주주총회자료준비하기', '기업설명회발표자료만들기', '기업인수합병검토하기', 
    '기업공개를위한서류준비하기',

    # 문화 및 엔터테인먼트
    '좋아하는가수콘서트예매하기', '미술관에서전시회관람하기', '브로드웨이뮤지컬티켓구매하기', '지역극단연극보러가기', 
    '기대되는영화개봉일체크하기', '인기예능프로그램본방사수하기', '팟캐스트라디오쇼청취하기', '아이돌팬미팅참석하기', 
    '좋아하는작가사인회가기', '유명인사초청북콘서트참여하기', '시사토크콘서트청강하기',

    # 사회 및 정치 관련
    '지방선거후보공약살펴보기', '국회본회의생중계시청하기', '새정부정책발표회참석하기', '국제외교회담결과분석하기', 
    '남북정상회담중계보기', '대통령기자회견시청하기', '주요이슈에대한여론조사참여하기', '지역시민단체활동참가하기', 
    '주말봉사활동참여하기', '연말자선단체에기부하기',

    # 환경 및 지속가능성 관련
    '분리수거철저히하기', '일회용품사용줄이기', '탄소중립생활실천하기', '에너지절약용제품사용하기', '멸종위기동물보호캠페인참여하기', 
    '기후변화대응정책제안하기', '지속가능한소비습관기르기', '환경교육프로그램참가하기', '그린캠페인에동참하기',

    # 금융 및 경제 관련
    '주식시장동향체크하기', '안정적인펀드상품가입하기', '부동산시장분석후매물검토하기', '외환시장변동추이파악하기', 
    '중앙은행금리인상영향분석하기', '물가상승률체감하기', '월간경제지표발표확인하기', '5월종합소득세신고하기', 
    '연말정산서류준비하기', '내년도가계예산편성하기',

    # 기타
    '지진대비긴급대피훈련참가하기', '화재발생시대피요령숙지하기', '직장내재난대비모의훈련실시하기', '안전교육이수증발급받기', 
    '정기소방훈련참여하기', '응급처치기본과정수료하기', '실시간교통정보확인후출발하기', '주간날씨예보체크하기', 
    '공휴일맞춰여행계획세우기', '가족과함께기념일축하하기', '지역축제참여해문화체험하기', '학교축제준비위원회활동하기'
])
# DEPARTMENTS, SUBJECTS, PROFESSORS, BUILDINGS, ROOM_TYPES, TIMES, CLASS_TYPES, ACADEMIC_TERMS, EVENTS에도 추가
DEPARTMENTS.extend(['첨단인공지능융합학과', '스마트로봇공학과', '지속가능환경공학과', '미래식품영양학과', '글로벌관광경영학과', '융합미디어커뮤니케이션학과'])
SUBJECTS.extend(['인공지능윤리와사회적책임', '지속가능발전을위한혁신기술', '창의적문제해결과디자인씽킹', '글로벌이슈와리더십개발', '다학제간융합캡스톤디자인프로젝트'])
PROFESSORS.extend(['남궁', '심우철', '노정호', '정약용', '하상욱', '곽진', '성삼문', '차미리사', '주시경', '우장춘', '구인회'])
BUILDINGS.extend(['미래형스마트팩토리', '창의융합교육센터', '글로벌문화교류라운지', '첨단기술메이커스페이스', '창업혁신지원관'])
ROOM_TYPES.extend(['가상현실체험실습실', '3D프린팅제작소', '오픈이노베이션코워킹스페이스', '글로벌화상회의실', '멀티미디어콘텐츠제작실'])
TIMES.extend(['07:30-09:00', '21:00-22:30', '22:30-24:00'])
CLASS_TYPES.extend(['거꾸로학습(플립러닝)', '협력적팀티칭수업', '온오프라인블렌디드러닝', '대규모공개온라인강좌(MOOC)', '문제중심학습(PBL)'])
ACADEMIC_TERMS.extend(['학점은행제를통한학위취득', '자기주도적독립연구프로그램', '학생설계융합전공', '다학제간융합전공과정', '단기집중이수마이크로디그리'])
EVENTS.extend(['캠퍼스혁신창업아이디어경진대회', '국제학술심포지엄및글로벌포럼', '산학협력기술혁신엑스포', '인공지능알고리즘경진대회', '지속가능캠퍼스그린캠페인'])

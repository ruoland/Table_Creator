from sqlalchemy.sql.sqltypes import DateTime
from sqlalchemy import create_engine,  String, ForeignKey, Text, Time, Date
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from datetime import time
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
# 데이터베이스 연결

engine = create_engine(DATABASE_URL)
SessionFactory = sessionmaker(bind=engine)

Base = declarative_base()


SessionFactory = sessionmaker(bind=engine)

@contextmanager
def session_scope():
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# 사용자 개인 정보: 유저 ID, 이메일과 비밀번호, 이름, 휴대폰 번호, 가입 날짜, 생일
# 다이어리 정보 : 유저 ID, 일기 ID, 일기 제목, 내용, 감정 상태, 날씨, 작성 날짜, 태그, 이미지
# 일정 정보 : 유저 ID, 날짜, 시작 시간, 마감 시간, 반복 주기 여부, 위치, 요일, 일정 제목, 일정 내용, 알림 시간, 알림 여부
# 구성원 정보 : 유저 ID, 구성원 목록
# 감정 정보 : 유저 ID, 감정 데이터, 감정 날짜, 키워드 정보, 키워드 날짜

# AI 정보 : 일기 분석 정보, 일기 태그 부여, 
# 모델 정의
class User(Base):
    __tablename__ = 'users'
    #기본 키
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(60), nullable=False)
    phone_number: Mapped[str] = mapped_column(unique=True, nullable=False)
    birth_day: Mapped[str] = mapped_column(default='생일 없음', nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    diaries: Mapped[list["Diary"]] = relationship("Diary", back_populates="user")
    schedules: Mapped[list["Schedule"]] = relationship('Schedule', back_populates='user')
    checklists: Mapped[list["Checklist"]] = relationship('Checklist', back_populates='user')
    hobbies: Mapped[list["Hobby"]] = relationship("Hobby", back_populates="user")
    gender: Mapped[str] = mapped_column(String(10), nullable=True)
class Diary(Base):
    __tablename__ = 'diaries'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)
    write_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    subtitle: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    weather: Mapped[str] = mapped_column(nullable=True)
    tags: Mapped[str] = mapped_column(nullable=True)
    emotion_data: Mapped[str] = mapped_column(nullable=True)
    keyword: Mapped[str] = mapped_column(nullable=True)
    user: Mapped["User"] = relationship("User", back_populates="diaries")
    images: Mapped[list["Image"]] = relationship("Image", back_populates="diary")

class Image(Base):
    __tablename__ = 'images'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    diary_id: Mapped[int] = mapped_column(ForeignKey('diaries.id'), nullable=False)
    file_path: Mapped[str] = mapped_column(nullable=False)
    file_name: Mapped[str] = mapped_column(nullable=False)
    diary: Mapped["Diary"] = relationship("Diary", back_populates="images")

class Checklist(Base):
    __tablename__ = 'checklist'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)
    write_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    deadline: Mapped[datetime] = mapped_column(nullable=False) 
    is_complete: Mapped[bool] = mapped_column(default=False, nullable=False)
    complete_date : Mapped[datetime] = mapped_column(DateTime, nullable=True)
    category: Mapped[str] = mapped_column(nullable=True)
    divisible: Mapped[bool] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="checklists")

class Hobby(Base):
    __tablename__ = 'hobby'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)
    name : Mapped[str] = mapped_column(nullable=False)
    preferred_time_start : Mapped[time] = mapped_column(Time, nullable=True) # 자동 추천 선호 시간
    preferred_time_end : Mapped[time] = mapped_column(Time, nullable=True) #선호 시간
    count : Mapped[int] = mapped_column(nullable=True)
    play_time : Mapped[float] = mapped_column(nullable=False) # 보통 소요 시간
    last_done_date : Mapped[datetime] = mapped_column(nullable=True) # 마지막으로 이 취미를 한 날
    user: Mapped["User"] = relationship("User", back_populates="hobbies")
class Schedule(Base):
    __tablename__ = 'schedule'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)
    date: Mapped[datetime] = mapped_column(nullable=False)
    subtitle: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=True)
    start_time: Mapped[datetime] = mapped_column(default=func.now(), nullable=False)
    end_time: Mapped[datetime] = mapped_column(nullable=True)
    is_loop: Mapped[bool] = mapped_column(default=False, nullable=False)
    location: Mapped[str] = mapped_column(default='위치 없음', nullable=False)
    is_alarm: Mapped[bool] = mapped_column(default=True, nullable=False)
    alarm_time: Mapped[datetime] = mapped_column(nullable=True)
    user: Mapped["User"] = relationship("User", back_populates="schedules")
    
class Holiday(Base):
    __tablename__ = 'anniversaries'

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(nullable=False)
    is_holiday: Mapped[bool] = mapped_column(nullable=False)
    
    def __repr__(self):
        return f"<Holiday(date={self.date}, name='{self.name}', is_public_holiday={self.is_public_holiday})>"

    
# 데이터베이스 테이블 생성
# 최초 1회만
from sqlalchemy.orm import configure_mappers
configure_mappers()
Base.metadata.bind = engine
Base.metadata.create_all(bind=engine)


    

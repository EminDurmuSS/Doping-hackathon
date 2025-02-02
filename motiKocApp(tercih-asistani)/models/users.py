# models/users.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class User:
    id: Optional[int]
    username: str
    password: str  # This should be hashed
    name: str
    email: str
    grade: str
    city: Optional[str] = None
    target_university: Optional[str] = None
    target_department: Optional[str] = None
    target_rank: Optional[int] = None
    study_type: str = "TYT"
    created_at: str = datetime.now().strftime('%Y-%m-%d')

    @classmethod
    def from_db_row(cls, row: dict) -> 'User':
        return cls(
            id=row['id'],
            username=row['username'],
            password=row['password'],
            name=row['name'],
            email=row['email'],
            grade=row['grade'],
            city=row.get('city'),
            target_university=row.get('target_university'),
            target_department=row.get('target_department'),
            target_rank=row.get('target_rank'),
            study_type=row.get('study_type', 'TYT'),
            created_at=row.get('created_at', datetime.now().strftime('%Y-%m-%d'))
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'username': self.username,
            'name': self.name,
            'email': self.email,
            'grade': self.grade,
            'city': self.city,
            'target_university': self.target_university,
            'target_department': self.target_department,
            'target_rank': self.target_rank,
            'study_type': self.study_type,
            'created_at': self.created_at
        }

@dataclass
class UserLevel:
    id: Optional[int]
    user_id: int
    current_level: int = 1
    current_xp: int = 0
    total_xp: int = 0

    @classmethod
    def from_db_row(cls, row: dict) -> 'UserLevel':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            current_level=row['current_level'],
            current_xp=row['current_xp'],
            total_xp=row['total_xp']
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'current_level': self.current_level,
            'current_xp': self.current_xp,
            'total_xp': self.total_xp
        }

@dataclass
class UserProfile:
    id: Optional[int]
    user_id: int
    study_streak: int = 0
    total_study_time: int = 0
    total_questions_solved: int = 0
    achievements_count: int = 0
    friends_count: int = 0
    last_active: str = datetime.now().strftime('%Y-%m-%d')

    @classmethod
    def from_db_row(cls, row: dict) -> 'UserProfile':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            study_streak=row.get('study_streak', 0),
            total_study_time=row.get('total_study_time', 0),
            total_questions_solved=row.get('total_questions_solved', 0),
            achievements_count=row.get('achievements_count', 0),
            friends_count=row.get('friends_count', 0),
            last_active=row.get('last_active', datetime.now().strftime('%Y-%m-%d'))
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'study_streak': self.study_streak,
            'total_study_time': self.total_study_time,
            'total_questions_solved': self.total_questions_solved,
            'achievements_count': self.achievements_count,
            'friends_count': self.friends_count,
            'last_active': self.last_active
        }

@dataclass
class StudyLog:
    id: Optional[int]
    user_id: int
    subject: str
    topic: str
    duration: int
    date: str
    performance_rating: int
    notes: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: dict) -> 'StudyLog':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            subject=row['subject'],
            topic=row['topic'],
            duration=row['duration'],
            date=row['date'],
            performance_rating=row['performance_rating'],
            notes=row.get('notes')
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'subject': self.subject,
            'topic': self.topic,
            'duration': self.duration,
            'date': self.date,
            'performance_rating': self.performance_rating,
            'notes': self.notes
        }
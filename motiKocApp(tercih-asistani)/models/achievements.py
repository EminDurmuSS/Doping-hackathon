# models/achievements.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class Badge:
    id: Optional[int]
    name: str
    description: str
    icon: str
    category: str
    requirement_type: str
    requirement_value: int
    earned_date: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: dict) -> 'Badge':
        return cls(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            icon=row['icon'],
            category=row['category'],
            requirement_type=row['requirement_type'],
            requirement_value=row['requirement_value'],
            earned_date=row.get('earned_date')
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'category': self.category,
            'requirement_type': self.requirement_type,
            'requirement_value': self.requirement_value,
            'earned_date': self.earned_date
        }

@dataclass
class Achievement:
    id: Optional[int]
    user_id: int
    title: str
    description: str
    date: str
    shared: bool = False
    likes: int = 0

    @classmethod
    def from_db_row(cls, row: dict) -> 'Achievement':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            title=row['title'],
            description=row['description'],
            date=row['date'],
            shared=row.get('shared', False),
            likes=row.get('likes', 0)
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'description': self.description,
            'date': self.date,
            'shared': self.shared,
            'likes': self.likes
        }

@dataclass
class Goal:
    id: Optional[int]
    user_id: int
    title: str
    deadline: str
    progress: int
    completed: bool
    created_at: str = datetime.now().strftime('%Y-%m-%d')

    @classmethod
    def from_db_row(cls, row: dict) -> 'Goal':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            title=row['title'],
            deadline=row['deadline'],
            progress=row['progress'],
            completed=row['completed'],
            created_at=row.get('created_at', datetime.now().strftime('%Y-%m-%d'))
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'deadline': self.deadline,
            'progress': self.progress,
            'completed': self.completed,
            'created_at': self.created_at
        }

@dataclass
class DailyTask:
    id: Optional[int]
    user_id: int
    task_type: str
    description: str
    xp_reward: int
    completed: bool = False
    date_created: str = datetime.now().strftime('%Y-%m-%d')
    date_completed: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: dict) -> 'DailyTask':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            task_type=row['task_type'],
            description=row['description'],
            xp_reward=row['xp_reward'],
            completed=row['completed'],
            date_created=row['date_created'],
            date_completed=row.get('date_completed')
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'task_type': self.task_type,
            'description': self.description,
            'xp_reward': self.xp_reward,
            'completed': self.completed,
            'date_created': self.date_created,
            'date_completed': self.date_completed
        }
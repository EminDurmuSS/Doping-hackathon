# features/forum/models.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class ForumPost:
    id: Optional[int]
    user_id: int
    title: str
    content: str
    category_id: int
    created_at: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    updated_at: Optional[str] = None
    view_count: int = 0
    is_solved: bool = False
    
    @classmethod
    def from_db_row(cls, row: dict) -> 'ForumPost':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            title=row['title'],
            content=row['content'],
            category_id=row['category_id'],
            created_at=row['created_at'],
            updated_at=row.get('updated_at'),
            view_count=row['view_count'],
            is_solved=row['is_solved']
        )

@dataclass
class ForumComment:
    id: Optional[int]
    post_id: int
    user_id: int
    content: str
    created_at: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    updated_at: Optional[str] = None
    is_accepted: bool = False
    upvotes: int = 0
    
    @classmethod
    def from_db_row(cls, row: dict) -> 'ForumComment':
        return cls(
            id=row['id'],
            post_id=row['post_id'],
            user_id=row['user_id'],
            content=row['content'],
            created_at=row['created_at'],
            updated_at=row.get('updated_at'),
            is_accepted=row['is_accepted'],
            upvotes=row['upvotes']
        )

@dataclass
class ForumCategory:
    id: Optional[int]
    name: str
    description: str
    icon: str
    order_index: int
    
    @classmethod
    def from_db_row(cls, row: dict) -> 'ForumCategory':
        return cls(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            icon=row['icon'],
            order_index=row['order_index']
        )

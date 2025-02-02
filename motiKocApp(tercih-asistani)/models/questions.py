# models/questions.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

@dataclass
class Question:
    id: Optional[int]
    subject: str
    topic: str
    subtopic: Optional[str]
    difficulty: int
    content: str
    answer: str
    explanation: Optional[str] = None
    exam_type: str = "TYT"  # TYT or AYT
    created_at: str = datetime.now().strftime('%Y-%m-%d')

    @classmethod
    def from_db_row(cls, row: dict) -> 'Question':
        return cls(
            id=row['id'],
            subject=row['subject'],
            topic=row['topic'],
            subtopic=row.get('subtopic'),
            difficulty=row['difficulty'],
            content=row['content'],
            answer=row['answer'],
            explanation=row.get('explanation'),
            exam_type=row.get('exam_type', 'TYT'),
            created_at=row.get('created_at', datetime.now().strftime('%Y-%m-%d'))
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'subject': self.subject,
            'topic': self.topic,
            'subtopic': self.subtopic,
            'difficulty': self.difficulty,
            'content': self.content,
            'answer': self.answer,
            'explanation': self.explanation,
            'exam_type': self.exam_type,
            'created_at': self.created_at
        }

@dataclass
class QuestionStat:
    id: Optional[int]
    user_id: int
    subject: str
    topic: str
    correct: int
    incorrect: int
    unanswered: int
    average_time: int
    last_practice: str

    @classmethod
    def from_db_row(cls, row: dict) -> 'QuestionStat':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            subject=row['subject'],
            topic=row['topic'],
            correct=row['correct'],
            incorrect=row['incorrect'],
            unanswered=row['unanswered'],
            average_time=row['average_time'],
            last_practice=row['last_practice']
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'subject': self.subject,
            'topic': self.topic,
            'correct': self.correct,
            'incorrect': self.incorrect,
            'unanswered': self.unanswered,
            'average_time': self.average_time,
            'last_practice': self.last_practice
        }

@dataclass
class MockExam:
    id: Optional[int]
    user_id: int
    exam_type: str
    exam_date: str
    total_time: int
    subject_results: Dict[str, Any]
    analysis: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: dict) -> 'MockExam':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            exam_type=row['exam_type'],
            exam_date=row['exam_date'],
            total_time=row['total_time'],
            subject_results=row['subject_results'],
            analysis=row.get('analysis')
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'exam_type': self.exam_type,
            'exam_date': self.exam_date,
            'total_time': self.total_time,
            'subject_results': self.subject_results,
            'analysis': self.analysis
        }

@dataclass
class SavedSolution:
    id: Optional[int]
    user_id: int
    subject: str
    question: str
    solution: str
    date: str = datetime.now().strftime('%Y-%m-%d')

    @classmethod
    def from_db_row(cls, row: dict) -> 'SavedSolution':
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            subject=row['subject'],
            question=row['question'],
            solution=row['solution'],
            date=row['date']
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'subject': self.subject,
            'question': self.question,
            'solution': self.solution,
            'date': self.date
        }
import sqlite3
import threading
import queue
import logging
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
from datetime import datetime
from sqlite3 import Cursor  # Added import

from config.settings import DATABASE_NAME, DB_TIMEOUT, DB_PRAGMAS

from config.constants import (
    DEFAULT_SUBJECTS,
    DEFAULT_BADGES,
    FORUM_CATEGORIES,
    YKS_CATEGORIES,            # Dictionary
    YKS_CATEGORIES_LIST,       # List
    STUDY_TYPES,
)

# Enhanced logging configuration
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database errors"""
    pass

class DatabaseConnectionPool:
    """Thread-safe database connection pool"""
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, max_connections: int = 10):
        self.pool = queue.Queue(maxsize=max_connections)
        self._fill_pool()
    
    def _fill_pool(self):
        """Initialize connection pool"""
        while not self.pool.full():
            conn = self._create_connection()
            self.pool.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with proper configuration"""
        try:
            conn = sqlite3.connect(DATABASE_NAME, timeout=DB_TIMEOUT, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            
            # Configure connection
            conn.execute('PRAGMA journal_mode=WAL')
            for pragma, value in DB_PRAGMAS.items():
                conn.execute(f'PRAGMA {pragma}={value}')
            conn.execute('PRAGMA foreign_keys=ON')
            
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to create database connection: {e}")
            raise DatabaseError(f"Connection creation failed: {e}")
    
    @classmethod
    def get_instance(cls) -> 'DatabaseConnectionPool':
        """Get singleton instance of connection pool"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = DatabaseConnectionPool()
        return cls._instance
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool"""
        try:
            return self.pool.get(timeout=DB_TIMEOUT)
        except queue.Empty:
            logger.error("Connection pool timeout")
            raise DatabaseError("Failed to get database connection from pool")
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        try:
            self.pool.put(conn, timeout=DB_TIMEOUT)
        except queue.Full:
            logger.warning("Connection pool full, closing connection")
            conn.close()

@contextmanager
def db_transaction(retries: int = 3):
    """Enhanced context manager for database transactions with retry logic"""
    pool = DatabaseConnectionPool.get_instance()
    conn = None
    
    for attempt in range(retries):
        try:
            conn = pool.get_connection()
            yield conn
            conn.commit()
            break
        except sqlite3.OperationalError as e:
            if 'database is locked' in str(e) and attempt < retries - 1:
                logger.warning(f"Database locked, attempt {attempt + 1} of {retries}")
                if conn:
                    conn.rollback()
                continue
            raise DatabaseError(f"Transaction failed: {e}")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Transaction error: {e}")
            raise DatabaseError(f"Transaction failed: {e}")
        finally:
            if conn:
                pool.return_connection(conn)

def init_db():
    """Initialize database with enhanced error handling and logging"""
    try:
        with db_transaction() as conn:
            cursor = conn.cursor()
            
            # Create tables with proper indexing
            _create_tables(cursor)
            _create_indexes(cursor)
            _initialize_default_data(cursor)
            
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise DatabaseError(f"Database initialization failed: {e}")

def _create_tables(cursor: sqlite3.Cursor):
    """Create all database tables with proper constraints and relationships"""
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            grade TEXT NOT NULL,
            city TEXT,
            target_university TEXT,
            target_department TEXT,
            target_rank INTEGER,
            study_type TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Login attempts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            attempt_time TEXT NOT NULL,
            ip_address TEXT,
            success BOOLEAN DEFAULT FALSE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

# Add COALESCE to handle NULL values in study_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS study_logs (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT,
            duration INTEGER NOT NULL DEFAULT 0,
            date TEXT NOT NULL,
            performance_rating INTEGER DEFAULT 0 CHECK(performance_rating BETWEEN 0 AND 5),
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Mood logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mood_logs (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            mood TEXT NOT NULL,
            stress_level INTEGER CHECK(stress_level BETWEEN 1 AND 5),
            notes TEXT,
            date TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Goals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            deadline TEXT NOT NULL,
            progress INTEGER DEFAULT 0 CHECK(progress BETWEEN 0 AND 100),
            completed BOOLEAN DEFAULT FALSE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Subjects table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # YKS Subjects table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS yks_subjects (
            id INTEGER PRIMARY KEY,
            exam_type TEXT NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            subtopic TEXT,
            difficulty INTEGER CHECK(difficulty BETWEEN 1 AND 5),
            importance INTEGER CHECK(importance BETWEEN 1 AND 5),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Saved Solutions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_solutions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            question TEXT NOT NULL,
            solution TEXT NOT NULL,
            date TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Friendships table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS friendships (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            friend_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'accepted', 'rejected')),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(friend_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, friend_id)
        )
    ''')

    # Achievements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS achievements (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            date TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Study Groups table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS study_groups (
            id INTEGER PRIMARY KEY,
            creator_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            group_type TEXT NOT NULL,
            max_members INTEGER NOT NULL DEFAULT 10,
            description TEXT,
            subjects TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(creator_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Group Members table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS group_members (
            id INTEGER PRIMARY KEY,
            group_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL DEFAULT 'member' CHECK(role IN ('admin', 'member')),
            joined_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(group_id) REFERENCES study_groups(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(group_id, user_id)
        )
    ''')

    # Update the badges table schema to include points
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS badges (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            icon TEXT NOT NULL,
            category TEXT NOT NULL,
            requirement_type TEXT NOT NULL,
            requirement_value INTEGER NOT NULL,
            points INTEGER NOT NULL DEFAULT 100,  -- Added points column
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # User Badges table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_badges (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            badge_id INTEGER NOT NULL,
            earned_date TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(badge_id) REFERENCES badges(id) ON DELETE CASCADE,
            UNIQUE(user_id, badge_id)
        )
    ''')

    # User Levels table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_levels (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            current_level INTEGER NOT NULL DEFAULT 1,
            current_xp INTEGER NOT NULL DEFAULT 0,
            total_xp INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id)
        )
    ''')

    # Daily Tasks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_tasks (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            task_type TEXT NOT NULL,
            description TEXT NOT NULL,
            xp_reward INTEGER NOT NULL,
            completed BOOLEAN DEFAULT FALSE,
            date_created TEXT NOT NULL,
            date_completed TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Competitions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS competitions (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            competition_type TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'active', 'completed')),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Competition Participants table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS competition_participants (
            id INTEGER PRIMARY KEY,
            competition_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            score INTEGER DEFAULT 0,
            joined_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(competition_id, user_id)
        )
    ''')

    # Mock Exams table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mock_exams (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            exam_type TEXT NOT NULL,
            exam_date TEXT NOT NULL,
            total_time INTEGER NOT NULL,
            subject_results TEXT NOT NULL, -- JSON formatted
            analysis TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Question Stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS question_stats (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            correct INTEGER DEFAULT 0,
            incorrect INTEGER DEFAULT 0,
            unanswered INTEGER DEFAULT 0,
            average_time INTEGER DEFAULT 0,
            last_practice TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Forum Categories table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forum_categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            icon TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Forum Questions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forum_questions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            view_count INTEGER DEFAULT 0,
            is_solved BOOLEAN DEFAULT FALSE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(category_id) REFERENCES forum_categories(id) ON DELETE CASCADE
        )
    ''')

    # Forum Answers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forum_answers (
            id INTEGER PRIMARY KEY,
            question_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_accepted BOOLEAN DEFAULT FALSE,
            upvotes INTEGER DEFAULT 0,
            FOREIGN KEY(question_id) REFERENCES forum_questions(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # XP Bonuses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS xp_bonuses (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            amount INTEGER NOT NULL,
            reason TEXT NOT NULL,
            awarded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Achievement Progress table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS achievement_progress (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            achievement_id INTEGER NOT NULL,
            current_value INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(achievement_id) REFERENCES badges(id) ON DELETE CASCADE,
            UNIQUE(user_id, achievement_id)
        )
    ''')

    # Study Preferences table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS study_preferences (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            daily_hours INTEGER NOT NULL DEFAULT 3,
            daily_questions INTEGER NOT NULL DEFAULT 50,
            start_time TEXT NOT NULL DEFAULT '09:00',
            end_time TEXT NOT NULL DEFAULT '22:00',
            break_duration INTEGER NOT NULL DEFAULT 15,
            study_duration INTEGER NOT NULL DEFAULT 45,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id)
        )
    ''')
    

def _initialize_default_data(cursor: Cursor) -> None:
    """Initialize database with default data."""
    try:
        # Initialize default subjects
        cursor.executemany(
            'INSERT OR IGNORE INTO subjects (name, category) VALUES (?, ?)',
            DEFAULT_SUBJECTS
        )
        logger.info(f"Initialized {len(DEFAULT_SUBJECTS)} default subjects")

        # Initialize default badges
        cursor.executemany('''
            INSERT OR IGNORE INTO badges 
            (name, description, icon, category, requirement_type, requirement_value)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', DEFAULT_BADGES)
        logger.info(f"Initialized {len(DEFAULT_BADGES)} default badges")

        # Initialize forum categories
        cursor.executemany('''
            INSERT OR IGNORE INTO forum_categories 
            (name, description, icon, order_index)
            VALUES (?, ?, ?, ?)
        ''', FORUM_CATEGORIES)
        logger.info(f"Initialized {len(FORUM_CATEGORIES)} forum categories")

        # Initialize YKS subjects
        for exam_type, categories in YKS_CATEGORIES.items():  # Ensure YKS_CATEGORIES is a dict
            if isinstance(categories, dict):
                for category, subjects in categories.items():
                    for subject in subjects:
                        cursor.execute('''
                            INSERT OR IGNORE INTO yks_subjects 
                            (exam_type, subject, topic, difficulty, importance)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (exam_type, subject, category, 3, 3))
            else:
                for subject in categories:
                    cursor.execute('''
                        INSERT OR IGNORE INTO yks_subjects 
                        (exam_type, subject, topic, difficulty, importance)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (exam_type, subject, "Genel", 3, 3))
        logger.info("Initialized YKS subjects")

    except sqlite3.Error as e:
        logger.error(f"Error initializing default data: {e}")
        raise DatabaseError(f"Failed to initialize default data: {e}")

def _create_indexes(cursor: sqlite3.Cursor):
    """Create optimized indexes"""
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_study_logs_user_date ON study_logs(user_id, date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_achievements_user ON achievements(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mock_exams_user_date ON mock_exams(user_id, exam_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_forum_questions_user ON forum_questions(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_forum_answers_user ON forum_answers(user_id)')

class DatabaseManager:
    """High-level database operations manager"""
    
    @staticmethod
    @contextmanager
    def get_cursor():
        """Get database cursor with automatic cleanup"""
        with db_transaction() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
    
    @staticmethod
    def execute_query(query: str, params: tuple = (), fetch_all: bool = False) -> Union[List[sqlite3.Row], Optional[sqlite3.Row]]:
        """Execute database query with proper error handling"""
        try:
            with DatabaseManager.get_cursor() as cursor:
                cursor.execute(query, params)
                if fetch_all:
                    return cursor.fetchall()
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query execution failed: {e}")
    
    @staticmethod
    def insert_record(table: str, data: Dict[str, Any]) -> int:
        """Insert record with automatic ID generation"""
        placeholders = ', '.join(['?' for _ in data])
        columns = ', '.join(data.keys())
        query = f'INSERT INTO {table} ({columns}) VALUES ({placeholders})'
        
        try:
            with DatabaseManager.get_cursor() as cursor:
                cursor.execute(query, tuple(data.values()))
                return cursor.lastrowid or 0  # Or use another default value

        except Exception as e:
            logger.error(f"Insert operation failed: {e}")
            raise DatabaseError(f"Insert operation failed: {e}")
    
    @staticmethod
    def update_record(table: str, data: Dict[str, Any], condition: str, params: tuple) -> bool:
        """Update record with condition"""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f'UPDATE {table} SET {set_clause} WHERE {condition}'
        
        try:
            with DatabaseManager.get_cursor() as cursor:
                cursor.execute(query, tuple(data.values()) + params)
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Update operation failed: {e}")
            raise DatabaseError(f"Update operation failed: {e}")

# Export the DatabaseManager for higher-level operations
__all__ = ['DatabaseManager', 'db_transaction', 'DatabaseError', 'init_db']

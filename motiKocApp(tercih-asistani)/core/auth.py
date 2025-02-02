# core/auth.py

import sqlite3
import hashlib
from datetime import datetime
import streamlit as st
from typing import Optional, Dict, Any
import logging

from config.settings import (
    PASSWORD_MIN_LENGTH, 
    MAX_LOGIN_ATTEMPTS, 
    LOGIN_COOLDOWN
)
from config.constants import ERROR_MESSAGES, SUCCESS_MESSAGES
from core.database import db_transaction  # Remove get_db_connection import
from services.gamification import GamificationService  # Updated import

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to the logger if not already added
if not logger.handlers:
    logger.addHandler(ch)


class AuthError(Exception):
    """Custom exception for authentication errors"""
    pass

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password: str) -> bool:
    """Validate password meets requirements"""
    if len(password) < PASSWORD_MIN_LENGTH:
        return False
    # Add more password validation rules as needed
    return True

# Initialize GamificationService
gamification_service = GamificationService()

def login_user(username: str, password: str) -> Optional[int]:
    """
    Authenticate user and return user_id if successful.

    Args:
        username (str): Username.
        password (str): Plain text password.

    Returns:
        Optional[int]: User ID if login successful, None otherwise.

    Raises:
        AuthError: If login fails or too many attempts.
    """
    try:
        with db_transaction() as conn:
            c = conn.cursor()
            # Check login attempts
            if _check_login_attempts(username, c):
                raise AuthError("Too many login attempts. Please try again later.")

            # Verify credentials
            c.execute('''
                SELECT id, password 
                FROM users 
                WHERE username = ?
            ''', (username,))
            result = c.fetchone()

            if result and result['password'] == hash_password(password):
                _reset_login_attempts(username, c)
                return result['id']

            # Record failed attempt
            _record_login_attempt(username, c)
            return None

    except AuthError as e:
        raise
    except Exception as e:
        raise AuthError(f"An unexpected error occurred during login: {str(e)}")
def register_user(user_data: Dict[str, Any]) -> int:
    """
    Register a new user with improved transaction handling.
    """
    required_fields = [
        'username', 'password', 'name', 'email', 
        'grade', 'study_type'
    ]

    # Validate required fields
    for field in required_fields:
        if field not in user_data or not user_data[field]:
            raise AuthError(f"Missing required field: {field}")

    # Validate password
    if not validate_password(user_data['password']):
        raise AuthError(
            f"Password must be at least {PASSWORD_MIN_LENGTH} characters long"
        )

    try:
        with db_transaction() as conn:
            c = conn.cursor()

            # Check if username exists
            c.execute(
                'SELECT 1 FROM users WHERE username = ?', 
                (user_data['username'],)
            )
            if c.fetchone():
                raise AuthError("Username already exists")

            # Hash password
            hashed_password = hash_password(user_data['password'])

            # Insert user
            c.execute('''
                INSERT INTO users (
                    username, password, name, email, grade, city,
                    target_university, target_department, target_rank, study_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_data['username'], 
                hashed_password,
                user_data['name'],
                user_data['email'],
                user_data['grade'],
                user_data.get('city', ''),
                user_data.get('target_university', ''),
                user_data.get('target_department', ''),
                user_data.get('target_rank', 0),
                user_data['study_type']
            ))

            user_id = c.lastrowid
            if not user_id:
                raise AuthError("Failed to retrieve user ID after registration.")

            # Initialize user levels
            c.execute('''
                INSERT INTO user_levels (user_id, current_level, current_xp, total_xp)
                VALUES (?, 1, 0, 0)
            ''', (user_id,))

            # Check and award welcome badge with fallback XP value
            c.execute('''
                SELECT id, COALESCE(points, 100) as badge_points
                FROM badges 
                WHERE requirement_type = 'registration' 
                LIMIT 1
            ''')
            welcome_badge = c.fetchone()
            
            if welcome_badge:
                current_date = datetime.now().strftime('%Y-%m-%d')
                # Award badge
                c.execute('''
                    INSERT INTO user_badges (user_id, badge_id, earned_date)
                    VALUES (?, ?, ?)
                ''', (user_id, welcome_badge['id'], current_date))
                
                # Award XP with default value if points column doesn't exist
                xp_amount = welcome_badge['badge_points'] if welcome_badge['badge_points'] else 100
                
                c.execute('''
                    UPDATE user_levels
                    SET current_xp = current_xp + ?,
                        total_xp = total_xp + ?
                    WHERE user_id = ?
                ''', (xp_amount, xp_amount, user_id))

            return user_id

    except sqlite3.IntegrityError as e:
        logger.error(f"Database integrity error during registration: {e}")
        raise AuthError(f"Registration failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise AuthError(f"An unexpected error occurred during registration: {str(e)}")

def _check_login_attempts(username: str, cursor: sqlite3.Cursor) -> bool:
    """
    Check if user has exceeded maximum login attempts.

    Args:
        username (str): Username.
        cursor (sqlite3.Cursor): Database cursor.

    Returns:
        bool: True if maximum attempts exceeded, False otherwise.
    """
    try:
        cursor.execute('''
            SELECT COUNT(*) as attempt_count, 
                   MAX(attempt_time) as last_attempt
            FROM login_attempts
            WHERE username = ?
            AND attempt_time > datetime('now', ?)
        ''', (username, f'-{LOGIN_COOLDOWN} seconds'))

        result = cursor.fetchone()
        if result:
            logger.debug(f"User {username} has {result['attempt_count']} login attempts in the last {LOGIN_COOLDOWN} seconds.")
            return result['attempt_count'] >= MAX_LOGIN_ATTEMPTS
        return False
    except Exception as e:
        logger.error(f"Error checking login attempts for username '{username}': {e}")
        raise AuthError("Failed to check login attempts.")

def _record_login_attempt(username: str, cursor: sqlite3.Cursor):
    """
    Record a failed login attempt.

    Args:
        username (str): Username.
        cursor (sqlite3.Cursor): Database cursor.
    """
    try:
        cursor.execute('''
            INSERT INTO login_attempts (username, attempt_time)
            VALUES (?, datetime('now'))
        ''', (username,))
        logger.debug(f"Recorded failed login attempt for username '{username}'.")
    except Exception as e:
        logger.error(f"Error recording login attempt for username '{username}': {e}")
        raise AuthError("Failed to record login attempt.")

def _reset_login_attempts(username: str, cursor: sqlite3.Cursor):
    """
    Reset login attempts after successful login.

    Args:
        username (str): Username.
        cursor (sqlite3.Cursor): Database cursor.
    """
    try:
        cursor.execute('''
            DELETE FROM login_attempts
            WHERE username = ?
        ''', (username,))
        logger.debug(f"Reset login attempts for username '{username}'.")
    except Exception as e:
        logger.error(f"Error resetting login attempts for username '{username}': {e}")
        raise AuthError("Failed to reset login attempts.")

def check_auth() -> bool:
    """
    Check if user is authenticated.

    Returns:
        bool: True if user is authenticated, False otherwise.
    """
    return 'user_id' in st.session_state and st.session_state.user_id is not None

def logout_user():
    """
    Log out current user.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    logger.info("User logged out successfully.")
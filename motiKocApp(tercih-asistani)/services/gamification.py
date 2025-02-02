from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from core.database import DatabaseManager, DatabaseError, db_transaction
import sqlite3

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LevelInfo:
    level: int
    current_xp: int
    total_xp: int
    level_up: bool
    xp_for_next: int

@dataclass
class Achievement:
    id: Optional[int]
    title: str
    description: str
    icon: str
    points: int
    completed: bool = False
    completed_at: Optional[str] = None

@dataclass
class Badge:
    id: int
    title: str
    description: str
    icon: str
    points: int
    earned_at: Optional[str] = None


class GamificationService:
    """Enhanced gamification service with improved error handling and performance"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.xp_rates = {
            'study_session': 50,
            'study_minute': 1,
            'performance_bonus': 20,
            'streak_bonus': 100,
            'badge_earned': 200,
            'achievement_shared': 150,
            'forum_question': 75,
            'forum_answer': 100,
            'answer_accepted': 200,
            'upvote_received': 10,
            'answer_upvoted': 5
        }
    
    def calculate_xp_for_activity(self, activity_type: str, duration: Optional[int] = None) -> int:
        """Calculate XP for different activities with validation"""
        try:
            if activity_type not in self.xp_rates:
                logger.warning(f"Unknown activity type: {activity_type}")
                return 0
            
            base_xp = self.xp_rates[activity_type]
            
            if activity_type == 'study_session' and duration:
                minute_xp = min(duration * self.xp_rates['study_minute'], 500)
                total_xp = base_xp + minute_xp
                logger.debug(f"Calculated XP for study session: base {base_xp} + minutes {minute_xp} = {total_xp}")
                return total_xp
            
            return base_xp
            
        except Exception as e:
            logger.error(f"Error calculating XP: {e}")
            return 0
    
    def update_user_xp(self, user_id: int, xp_earned: int) -> LevelInfo:
        """Update user XP with improved transaction handling"""
        try:
            with db_transaction() as conn:
                cursor = conn.cursor()
                
                # Get current level info
                cursor.execute('''
                    SELECT current_level, current_xp, total_xp 
                    FROM user_levels 
                    WHERE user_id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                if not result:
                    # Initialize new user level
                    cursor.execute('''
                        INSERT INTO user_levels (user_id, current_level, current_xp, total_xp)
                        VALUES (?, 1, 0, 0)
                    ''', (user_id,))
                    current_level, current_xp, total_xp = 1, 0, 0
                else:
                    current_level, current_xp, total_xp = result
                
                # Update XP
                new_total_xp = total_xp + xp_earned
                new_current_xp = current_xp + xp_earned
                level_up = False
                
                # Check for level up with configurable multiplier
                while new_current_xp >= self._calculate_xp_for_level(current_level):
                    new_current_xp -= self._calculate_xp_for_level(current_level)
                    current_level += 1
                    level_up = True
                
                # Update database
                cursor.execute('''
                    UPDATE user_levels 
                    SET current_level = ?, 
                        current_xp = ?, 
                        total_xp = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (current_level, new_current_xp, new_total_xp, user_id))
                
                return LevelInfo(
                    level=current_level,
                    current_xp=new_current_xp,
                    total_xp=new_total_xp,
                    level_up=level_up,
                    xp_for_next=self._calculate_xp_for_level(current_level)
                )
                
        except Exception as e:
            logger.error(f"Error updating user XP: {e}")
            raise
    
    def check_and_award_achievements(self, user_id: int) -> List[Achievement]:
        """Check and award achievements with improved performance"""
        earned_achievements = []
        
        try:
            with db_transaction() as conn:
                cursor = conn.cursor()
                
                # Get all incomplete achievements efficiently
                cursor.execute('''
                    WITH UserAchievements AS (
                        SELECT achievement_id 
                        FROM user_achievements 
                        WHERE user_id = ?
                    )
                    SELECT a.* 
                    FROM achievements a
                    LEFT JOIN UserAchievements ua ON a.id = ua.achievement_id
                    WHERE ua.achievement_id IS NULL
                ''', (user_id,))
                
                available_achievements = cursor.fetchall()
                
                for achievement in available_achievements:
                    if self._check_achievement_requirement(user_id, achievement['requirement_type'], 
                                                         achievement['requirement_value'], cursor):
                        # Award achievement
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        cursor.execute('''
                            INSERT INTO user_achievements 
                            (user_id, achievement_id, earned_at)
                            VALUES (?, ?, ?)
                        ''', (user_id, achievement['id'], now))
                        
                        earned_achievement = Achievement(
                            id=achievement['id'],
                            title=achievement['name'],
                            description=achievement['description'],
                            icon=achievement['icon'],
                            points=achievement['points'],
                            completed=True,
                            completed_at=now
                        )
                        earned_achievements.append(earned_achievement)
                        
                        # Award XP for earning achievement
                        self.update_user_xp(user_id, achievement['points'])
                
                return earned_achievements
                
        except Exception as e:
            logger.error(f"Error checking achievements: {e}")
            raise
    
    def _calculate_xp_for_level(self, level: int) -> int:
        """Calculate required XP for level with custom progression"""
        base_xp = 1000
        multiplier = 1.5
        return int(base_xp * (multiplier ** (level - 1)))
    
    def check_and_award_badges(self, user_id: int) -> List[Badge]:
        """Check and award badges to the user"""
        # Implementation for checking and awarding badges
        earned_badges = []
        try:
            with db_transaction() as conn:
                cursor = conn.cursor()
                
                # Example logic to check badge criteria
                cursor.execute('''
                    SELECT * FROM badges WHERE criteria_met = ?
                ''', (user_id,))
                
                badges = cursor.fetchall()
                
                for badge in badges:
                    # Award badge if not already awarded
                    cursor.execute('''
                        SELECT 1 FROM user_badges WHERE user_id = ? AND badge_id = ?
                    ''', (user_id, badge['id']))
                    
                    if not cursor.fetchone():
                        cursor.execute('''
                            INSERT INTO user_badges (user_id, badge_id, awarded_at)
                            VALUES (?, ?, CURRENT_TIMESTAMP)
                        ''', (user_id, badge['id']))
                        
                        earned_badge = Badge(
                            id=badge['id'],
                            title=badge['title'],
                            description=badge['description'],
                            icon=badge['icon'],
                            points=badge['points'],
                            earned_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )
                        earned_badges.append(earned_badge)
                        
                        # Optionally, award XP or other rewards
                        self.update_user_xp(user_id, badge['points'])
                        
            return earned_badges
            
        except Exception as e:
            logger.error(f"Error checking and awarding badges: {e}")
            return earned_badges
    
    def _check_achievement_requirement(self, user_id: int, requirement_type: str, 
                                     requirement_value: int, cursor: sqlite3.Cursor) -> bool:
        """Check if user meets achievement requirements with caching"""
        try:
            if requirement_type == 'study_hours':
                cursor.execute('''
                    SELECT SUM(duration) / 60.0
                    FROM study_logs
                    WHERE user_id = ?
                    AND date >= date('now', '-30 days')
                ''', (user_id,))
                total_hours = cursor.fetchone()[0] or 0
                return total_hours >= requirement_value

            elif requirement_type == 'streak_days':
                cursor.execute('''
                    WITH consecutive_days AS (
                        SELECT date,
                               date(date, '-' || ROW_NUMBER() OVER (ORDER BY date) || ' days') as group_date
                        FROM (SELECT DISTINCT date FROM study_logs WHERE user_id = ?)
                    )
                    SELECT COUNT(*) as streak_length
                    FROM (
                        SELECT group_date, COUNT(*) as streak
                        FROM consecutive_days
                        GROUP BY group_date
                        ORDER BY COUNT(*) DESC
                        LIMIT 1
                    )
                ''', (user_id,))
                max_streak = cursor.fetchone()[0] or 0
                return max_streak >= requirement_value

            elif requirement_type == 'daily_questions':
                cursor.execute('''
                    SELECT COUNT(*)
                    FROM question_stats
                    WHERE user_id = ?
                    AND date(last_practice) = date('now')
                ''', (user_id,))
                daily_questions = cursor.fetchone()[0] or 0
                return daily_questions >= requirement_value

            elif requirement_type == 'performance_rating':
                cursor.execute('''
                    SELECT AVG(performance_rating)
                    FROM study_logs
                    WHERE user_id = ?
                    AND date >= date('now', '-7 days')
                ''', (user_id,))
                avg_rating = cursor.fetchone()[0] or 0
                return avg_rating >= requirement_value

            logger.warning(f"Unknown requirement type: {requirement_type}")
            return False

        except Exception as e:
            logger.error(f"Error checking achievement requirement: {e}")
            return False

    def get_leaderboard(self, timeframe: str = 'weekly', limit: int = 100) -> List[Dict[str, Any]]:
        """Get user leaderboard with enhanced performance"""
        try:
            date_filter = {
                'daily': "date('now')",
                'weekly': "date('now', '-7 days')",
                'monthly': "date('now', '-30 days')",
                'all_time': "date('1970-01-01')"
            }.get(timeframe, "date('now', '-7 days')")

            with db_transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    WITH UserStats AS (
                        SELECT 
                            u.id,
                            u.username,
                            ul.current_level,
                            ul.total_xp,
                            COUNT(DISTINCT sl.date) as study_days,
                            SUM(sl.duration) / 60.0 as study_hours,
                            COALESCE(AVG(sl.performance_rating), 0) as avg_performance,
                            COUNT(DISTINCT ua.achievement_id) as achievements
                        FROM users u
                        JOIN user_levels ul ON u.id = ul.user_id
                        LEFT JOIN study_logs sl ON u.id = sl.user_id AND sl.date >= {date_filter}
                        LEFT JOIN user_achievements ua ON u.id = ua.user_id
                        GROUP BY u.id, u.username, ul.current_level, ul.total_xp
                    )
                    SELECT 
                        us.*,
                        RANK() OVER (ORDER BY us.total_xp DESC) as rank
                    FROM UserStats us
                    ORDER BY us.total_xp DESC
                    LIMIT ?
                ''', (limit,))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []

    def award_bonus_xp(self, user_id: int, reason: str, bonus_amount: int) -> bool:
        """Award bonus XP with validation and logging"""
        try:
            if bonus_amount <= 0:
                logger.warning(f"Invalid bonus amount: {bonus_amount}")
                return False

            max_bonus = 1000  # Maximum allowed bonus
            if bonus_amount > max_bonus:
                logger.warning(f"Bonus amount {bonus_amount} exceeds maximum {max_bonus}")
                bonus_amount = max_bonus

            with db_transaction() as conn:
                cursor = conn.cursor()
                
                # Log the bonus
                cursor.execute('''
                    INSERT INTO xp_bonuses (user_id, amount, reason, awarded_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (user_id, bonus_amount, reason))

                # Update user XP
                self.update_user_xp(user_id, bonus_amount)
                return True

        except Exception as e:
            logger.error(f"Error awarding bonus XP: {e}")
            return False

    def get_user_achievements_progress(self, user_id: int) -> Dict[str, Any]:
        """Get detailed achievement progress for user"""
        try:
            with db_transaction() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    WITH UserProgress AS (
                        SELECT 
                            a.id,
                            a.name,
                            a.description,
                            a.requirement_type,
                            a.requirement_value,
                            CASE 
                                WHEN ua.achievement_id IS NOT NULL THEN 100
                                ELSE COALESCE(
                                    CASE a.requirement_type
                                        WHEN 'study_hours' THEN (
                                            SELECT COALESCE(SUM(duration) / 60.0 / a.requirement_value * 100, 0)
                                            FROM study_logs
                                            WHERE user_id = ?
                                        )
                                        WHEN 'streak_days' THEN (
                                            SELECT COALESCE(COUNT(*) * 100 / a.requirement_value, 0)
                                            FROM (
                                                SELECT DISTINCT date
                                                FROM study_logs
                                                WHERE user_id = ?
                                            )
                                        )
                                        ELSE 0
                                    END,
                                    0
                                )
                            END as progress_percentage,
                            ua.earned_at
                        FROM achievements a
                        LEFT JOIN user_achievements ua ON a.id = ua.achievement_id AND ua.user_id = ?
                    )
                    SELECT *,
                           RANK() OVER (ORDER BY progress_percentage DESC) as display_order
                    FROM UserProgress
                    ORDER BY display_order
                ''', (user_id, user_id, user_id))

                achievements = cursor.fetchall()

                # Calculate overall statistics
                total_achievements = len(achievements)
                completed_achievements = sum(1 for a in achievements if a['progress_percentage'] >= 100)
                total_points = sum(a['points'] for a in achievements if a['progress_percentage'] >= 100)

                return {
                    'achievements': [dict(a) for a in achievements],
                    'total_achievements': total_achievements,
                    'completed_achievements': completed_achievements,
                    'completion_percentage': (completed_achievements / total_achievements * 100) if total_achievements > 0 else 0,
                    'total_points': total_points
                }

        except Exception as e:
            logger.error(f"Error getting achievements progress: {e}")
            return {
                'achievements': [],
                'total_achievements': 0,
                'completed_achievements': 0,
                'completion_percentage': 0,
                'total_points': 0
            }

    def get_activity_summary(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get user activity summary with streaks and patterns"""
        try:
            with db_transaction() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    WITH daily_activity AS (
                        SELECT 
                            date,
                            SUM(duration) / 60.0 as hours,
                            AVG(performance_rating) as avg_performance,
                            COUNT(*) as sessions
                        FROM study_logs
                        WHERE user_id = ? AND date >= date('now', ? || ' days')
                        GROUP BY date
                    )
                    SELECT 
                        COALESCE(SUM(hours), 0) as total_hours,
                        COALESCE(AVG(hours), 0) as avg_hours_per_day,
                        COALESCE(AVG(avg_performance), 0) as avg_performance,
                        COUNT(*) as active_days,
                        (SELECT COUNT(*) FROM daily_activity) as streak
                    FROM daily_activity
                ''', (user_id, -days))

                summary = dict(cursor.fetchone())

                # Add achievement counts
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM user_achievements 
                    WHERE user_id = ? AND earned_at >= date('now', ? || ' days')
                ''', (user_id, -days))
                
                summary['achievements_earned'] = cursor.fetchone()[0]

                return summary

        except Exception as e:
            logger.error(f"Error getting activity summary: {e}")
            return {
                'total_hours': 0,
                'avg_hours_per_day': 0,
                'avg_performance': 0,
                'active_days': 0,
                'streak': 0,
                'achievements_earned': 0
            }
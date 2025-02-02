import streamlit as st
from typing import Optional, Dict, Any, List, TypeVar, Union, cast
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd
from sqlite3 import Row
from dataclasses import dataclass

from ..components.cards import Card, AlertCard, StatCard
from services.ai_service import AIService
from services.gamification import GamificationService, Achievement, LevelInfo
from core.database import DatabaseManager, db_transaction, DatabaseError

T = TypeVar('T', bound=Dict[str, Any])

@dataclass
class TaskInfo:
    id: int
    task_type: str
    description: str
    xp_reward: int
    completed: bool
    date_created: str
    date_completed: Optional[str] = None

class HomePage:
    def __init__(self, user_id: Optional[int] = None):
        self.user_id = user_id
        self.ai_service = AIService()
        self.gamification_service = GamificationService()
        self.db_manager = DatabaseManager()
    
    def _row_to_dict(self, row: Optional[Row]) -> Dict[str, Any]:
        """Convert sqlite3.Row to a dictionary."""
        if row is None:
            return {}
        return {key: row[key] for key in row.keys()} if isinstance(row, Row) else {}

    def _rows_to_dict_list(self, rows: Optional[List[Row]]) -> List[Dict[str, Any]]:
        """Convert a list of sqlite3.Row objects to a list of dictionaries."""
        if not rows:
            return []
        return [self._row_to_dict(row) for row in rows if isinstance(row, Row)]

    def _safe_get(self, data: Union[Row, Dict[str, Any], None], key: str, default: Any = None) -> Any:
        """Safely get a value from either a Row or Dict object."""
        if data is None:
            return default
        try:
            return data[key] if key in data else default
        except (KeyError, TypeError):
            return default

    def show(self):
        """Display home page content"""
        self._show_header()
        self._show_main_content()
        self._show_quick_actions()

    def _show_header(self):
        """Display page header with YKS countdown"""
        st.title(f"Hoş Geldin, {st.session_state.get('username', '')}! 👋")
        
        exam_date = datetime(2025, 6, 15)
        days_left = (exam_date - datetime.now()).days
        
        st.markdown(f"""
        <div class='custom-card animate-fade-in'>
            <h2>⏳ YKS'ye Kalan Süre</h2>
            <h1 style='color: var(--accent-color); text-align: center;'>
                {days_left} GÜN
            </h1>
            <p style='text-align: center;'>
                TYT: {exam_date.strftime('%d.%m.%Y')} Cumartesi<br>
                AYT: {(exam_date + timedelta(days=1)).strftime('%d.%m.%Y')} Pazar
            </p>
        </div>
        """, unsafe_allow_html=True)

    def _show_main_content(self):
        """Display main content in columns"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._show_study_summary()
            self._show_daily_tasks()
            self._show_performance_analytics()
        
        with col2:
            self._show_profile_summary()
            self._show_motivation_message()
            self._show_achievements()
            self._show_level_progress()

    def _get_study_data(self) -> Dict[str, Any]:
        """Get user's study data from the database."""
        try:
            study_stats_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT 
                    COUNT(DISTINCT date) as study_days,
                    COALESCE(SUM(duration), 0) as total_minutes,
                    COALESCE(AVG(duration), 0) as avg_duration
                FROM study_logs 
                WHERE user_id = ?
            ''', (self.user_id,), fetch_all=False))
            
            study_stats = self._row_to_dict(study_stats_query) if study_stats_query else {}

            subject_distribution_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                SELECT subject, COALESCE(SUM(duration), 0) as duration
                FROM study_logs
                WHERE user_id = ?
                GROUP BY subject
            ''', (self.user_id,), fetch_all=True))
            
            subject_distribution = self._rows_to_dict_list(subject_distribution_query)

            return {
                'study_days': self._safe_get(study_stats, 'study_days', 0),
                'total_hours': self._safe_get(study_stats, 'total_minutes', 0) / 60 if self._safe_get(study_stats, 'total_minutes', 0) is not None else 0,
                'daily_average': self._safe_get(study_stats, 'avg_duration', 0) / 60 if self._safe_get(study_stats, 'avg_duration', 0) is not None else 0,
                'subject_distribution': [
                    {
                        'subject': self._safe_get(row, 'subject', 'Unknown'),
                        'duration': self._safe_get(row, 'duration', 0) / 60
                    }
                    for row in subject_distribution
                ]
            }
        except DatabaseError as e:
            st.error(f"Çalışma verileri alınamadı: {str(e)}")
            return {
                'study_days': 0,
                'total_hours': 0,
                'daily_average': 0,
                'subject_distribution': []
            }

    def _show_study_summary(self):
        """Display study summary section"""
        try:
            study_data = self._get_study_data()
            
            st.markdown("### 📚 Çalışma Özeti")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                StatCard.metric_card(
                    "Toplam Çalışma",
                    f"{study_data.get('total_hours', 0):.1f} saat",
                    None
                )
            with col2:
                StatCard.metric_card(
                    "Çalışılan Gün",
                    str(study_data.get('study_days', 0)),
                    None
                )
            with col3:
                StatCard.metric_card(
                    "Günlük Ortalama",
                    f"{study_data.get('daily_average', 0):.1f} saat",
                    None
                )
            
            # Only show chart if there is data
            if study_data.get('subject_distribution') and len(study_data['subject_distribution']) > 0:
                chart_data = pd.DataFrame(study_data['subject_distribution'])
                if not chart_data.empty:
                    fig = px.pie(
                        chart_data,
                        values='duration',
                        names='subject',
                        title='Ders Dağılımı'
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                AlertCard.info(
                    "Henüz çalışma kaydı bulunmuyor. Hadi çalışmaya başlayalım! 📚"
                )
        except Exception as e:
            st.error(f"Çalışma özeti gösterilirken bir hata oluştu: {str(e)}")

    def _get_daily_tasks(self) -> List[TaskInfo]:
        """Get user's daily tasks from database"""
        try:
            tasks_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                SELECT * FROM daily_tasks
                WHERE user_id = ? AND date_created = date('now')
                ORDER BY completed ASC, task_type ASC
            ''', (self.user_id,), fetch_all=True))
            
            if not tasks_query:
                return []
            
            return [
                TaskInfo(
                    id=self._safe_get(task, 'id', 0),
                    task_type=self._safe_get(task, 'task_type', ''),
                    description=self._safe_get(task, 'description', ''),
                    xp_reward=self._safe_get(task, 'xp_reward', 0),
                    completed=self._safe_get(task, 'completed', False),
                    date_created=self._safe_get(task, 'date_created', ''),
                    date_completed=self._safe_get(task, 'date_completed')
                )
                for task in tasks_query
            ]
            
        except DatabaseError as e:
            st.error(f"Günlük görevler alınamadı: {str(e)}")
            return []

    def _show_daily_tasks(self):
        """Display daily tasks section"""
        st.markdown("### 📋 Günlük Görevler")
        
        tasks = self._get_daily_tasks()
        completed_count = sum(1 for task in tasks if task.completed)
        total_tasks = len(tasks)
        
        if total_tasks > 0:
            progress = completed_count / total_tasks
            st.progress(progress)
            st.write(f"Tamamlanan: {completed_count}/{total_tasks}")
            
            for task in tasks:
                self._render_task_card(task)
        else:
            AlertCard.info("Bugün için görev bulunmuyor.")

    def _render_task_card(self, task: TaskInfo):
        """Render individual task card"""
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                title = f"{task.task_type}: {task.description}"
                if task.completed:
                    title = f"✅ {title}"
                st.markdown(f"##### {title}")
                st.write(f"🎯 XP Ödülü: {task.xp_reward}")
            
            with col2:
                if not task.completed:
                    if st.button("Tamamla", key=f"task_{task.id}"):
                        self._complete_task(task.id)

    def _complete_task(self, task_id: int):
        """Handle task completion with XP and achievement updates"""
        if not self.user_id:
            st.error("Kullanıcı oturumu bulunamadı.")
            return

        try:
            with db_transaction() as conn:
                # Get task details
                task_query = cast(Optional[Row], self.db_manager.execute_query('''
                    SELECT xp_reward, task_type, description
                    FROM daily_tasks
                    WHERE id = ? AND user_id = ? AND completed = FALSE
                ''', (task_id, self.user_id), fetch_all=False))

                task_dict = self._row_to_dict(task_query)
                if not task_dict:
                    AlertCard.warning("Bu görev zaten tamamlanmış veya size ait değil.")
                    return

                # Update task status
                self.db_manager.execute_query('''
                    UPDATE daily_tasks
                    SET completed = TRUE,
                        date_completed = datetime('now')
                    WHERE id = ?
                ''', (task_id,), fetch_all=False)

                # Award XP
                xp_reward = self._safe_get(task_dict, 'xp_reward', 0)
                level_info = self.gamification_service.update_user_xp(self.user_id, xp_reward)

                # Award achievement XP bonus
                self.gamification_service.award_bonus_xp(
                    self.user_id,
                    f"Görev tamamlama: {self._safe_get(task_dict, 'description', '')}", 
                    xp_reward
                )

                # Check for achievements
                earned_achievements = self.gamification_service.check_and_award_achievements(self.user_id)

                # Update UI
                AlertCard.success(f"Görev tamamlandı! +{xp_reward} XP kazandın!")

                if level_info.level_up:
                    st.balloons()
                    AlertCard.success(f"🎉 Yeni seviyeye ulaştın: {level_info.level}!")

                for achievement in earned_achievements:
                    st.balloons()
                    AlertCard.success(f"🏆 Yeni başarı kazandın: {achievement.title}")

                # Refresh page
                st.rerun()

        except DatabaseError as e:
            st.error(f"Görev tamamlanırken bir hata oluştu: {str(e)}")
        except Exception as e:
            st.error(f"Beklenmeyen bir hata oluştu: {str(e)}")

    def _get_performance_data(self) -> Dict[str, Any]:
        """Get user's performance data and question statistics"""
        try:
            current_stats_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT 
                    AVG(sl.performance_rating) as avg_performance,
                    SUM(sl.duration) as total_minutes,
                    COUNT(DISTINCT sl.date) as study_days,
                    SUM(qs.correct) as correct_questions,
                    SUM(qs.incorrect) as incorrect_questions
                FROM study_logs sl
                LEFT JOIN question_stats qs ON sl.user_id = qs.user_id 
                AND sl.date = qs.last_practice
                WHERE sl.user_id = ? 
                AND sl.date >= date('now', '-7 days')
            ''', (self.user_id,), fetch_all=False))

            previous_stats_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT 
                    AVG(sl.performance_rating) as avg_performance,
                    SUM(sl.duration) as total_minutes,
                    COUNT(DISTINCT sl.date) as study_days,
                    SUM(qs.correct) as correct_questions,
                    SUM(qs.incorrect) as incorrect_questions
                FROM study_logs sl
                LEFT JOIN question_stats qs ON sl.user_id = qs.user_id 
                AND sl.date = qs.last_practice
                WHERE sl.user_id = ? 
                AND sl.date >= date('now', '-14 days')
                AND sl.date < date('now', '-7 days')
            ''', (self.user_id,), fetch_all=False))

            current = self._row_to_dict(current_stats_query)
            previous = self._row_to_dict(previous_stats_query)

            return {
                'avg_performance': self._safe_get(current, 'avg_performance', 0),
                'performance_change': (
                    (self._safe_get(current, 'avg_performance', 0) or 0) - 
                    (self._safe_get(previous, 'avg_performance', 0) or 0)
                ),
                'weekly_hours': self._safe_get(current, 'total_minutes', 0) / 60 if self._safe_get(current, 'total_minutes', 0) is not None else 0,
                'hours_change': (
                    (self._safe_get(current, 'total_minutes', 0) or 0) - 
                    (self._safe_get(previous, 'total_minutes', 0) or 0)
                ) / 60 if self._safe_get(current, 'total_minutes', 0) is not None and self._safe_get(previous, 'total_minutes', 0) is not None else 0,
                'total_questions': (
                    self._safe_get(current, 'incorrect_questions', 0)
                ),
                'questions_change': (
                    self._safe_get(current, 'correct_questions', 0) - 
                    self._safe_get(previous, 'correct_questions', 0)
                )
            }
        except DatabaseError as e:
            st.error(f"Performans verileri alınamadı: {str(e)}")
            return {
                'avg_performance': 0,
                'performance_change': 0,
                'weekly_hours': 0,
                'hours_change': 0,
                'total_questions': 0,
                'questions_change': 0
            }
        

    def _get_question_stats(self) -> List[Dict[str, Any]]:
        """Get detailed question statistics by topic"""
        try:
            stats_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                SELECT 
                    subject,
                    topic,
                    SUM(correct) as correct,
                    SUM(incorrect) as incorrect,
                    AVG(average_time) as avg_time
                FROM question_stats
                WHERE user_id = ?
                GROUP BY subject, topic
                ORDER BY subject, topic
            ''', (self.user_id,), fetch_all=True))
            
            return self._rows_to_dict_list(stats_query)
        except DatabaseError as e:
            st.error(f"Soru istatistikleri alınamadı: {str(e)}")
            return []

    def _show_performance_analytics(self):
        """Display performance analytics section"""
        st.markdown("### 📊 Performans Analizi")
        
        performance_data = self._get_performance_data()
        question_stats = self._get_question_stats()
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            StatCard.metric_card(
                "Ortalama Performans",
                f"{performance_data['avg_performance']:.1f}/5",
                f"{performance_data['performance_change']:+.1f}"
            )
        with col2:
            StatCard.metric_card(
                "Haftalık Çalışma",
                f"{performance_data['weekly_hours']:.1f} saat",
                f"{performance_data['hours_change']:+.1f}"
            )
        with col3:
            StatCard.metric_card(
                "Çözülen Soru",
                str(performance_data['total_questions']),
                f"{performance_data['questions_change']:+d}"
            )
        
        # Question performance chart
        if question_stats:
            st.markdown("#### Konu Bazlı Soru Performansı")
            chart_data = pd.DataFrame(question_stats)
            fig = px.bar(
                chart_data,
                x='topic',
                y=['correct', 'incorrect'],
                title='Konu Bazlı Doğru/Yanlış Analizi',
                barmode='group'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

    def _get_profile_data(self) -> Dict[str, Any]:
        """Get user's profile data"""
        try:
            profile_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT 
                    grade, 
                    study_type, 
                    target_university,
                    target_department,
                    target_rank,
                    city
                FROM users
                WHERE id = ?
            ''', (self.user_id,), fetch_all=False))
            
            return self._row_to_dict(profile_query)
        except DatabaseError as e:
            st.error(f"Profil bilgileri alınamadı: {str(e)}")
            return {}

    def _get_study_preferences(self) -> Dict[str, Any]:
        """Get user's study preferences"""
        try:
            prefs_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT 
                    daily_hours,
                    daily_questions,
                    start_time,
                    end_time,
                    break_duration,
                    study_duration
                FROM study_preferences
                WHERE user_id = ?
            ''', (self.user_id,), fetch_all=False))
            
            return self._row_to_dict(prefs_query)
        except DatabaseError as e:
            st.error(f"Çalışma tercihleri alınamadı: {str(e)}")
            return {}

    def _show_profile_summary(self):
        """Display profile summary section"""
        profile_data = self._get_profile_data()
        
        st.markdown("### 👤 Profil Özeti")
        
        with st.container():
            st.write(f"🎓 Sınıf: {self._safe_get(profile_data, 'grade', '-')}")
            st.write(f"📚 Alan: {self._safe_get(profile_data, 'study_type', '-')}")
            
            if self._safe_get(profile_data, 'target_university'):
                st.write(f"🎯 Hedef: {profile_data['target_university']}")
                st.write(f"📝 Bölüm: {self._safe_get(profile_data, 'target_department', '-')}")
                
                if self._safe_get(profile_data, 'target_rank'):
                    st.write(f"🏅 Hedef Sıralama: {profile_data['target_rank']:,}")
            
            study_prefs = self._get_study_preferences()
            if study_prefs:
                st.markdown("#### 📅 Çalışma Planı")
                st.write(f"🕒 {study_prefs['start_time']} - {study_prefs['end_time']}")
                st.write(f"⏱️ {study_prefs['study_duration']} dk çalışma / {study_prefs['break_duration']} dk mola")

    def _get_user_stats(self) -> Dict[str, Any]:
        """Get comprehensive user statistics for AI motivation"""
        try:
            stats_query = cast(Optional[Row], self.db_manager.execute_query('''
                WITH recent_activity AS (
                    SELECT 
                        COUNT(DISTINCT sl.date) as study_streak,
                        SUM(sl.duration) as total_minutes,
                        AVG(sl.performance_rating) as avg_performance,
                        SUM(qs.correct) as total_correct,
                        SUM(qs.incorrect) as total_incorrect
                    FROM study_logs sl
                    LEFT JOIN question_stats qs ON sl.user_id = qs.user_id
                    WHERE sl.user_id = ?
                    AND sl.date >= date('now', '-30 days')
                ),
                goal_stats AS (
                    SELECT COUNT(*) as active_goals
                    FROM goals
                    WHERE user_id = ?
                    AND completed = FALSE
                ),
                level_info AS (
                    SELECT 
                        current_level,
                        current_xp
                    FROM user_levels
                    WHERE user_id = ?
                ),
                mock_exam_stats AS (
                    SELECT 
                        COUNT(*) as total_exams,
                        MAX(exam_date) as last_exam
                    FROM mock_exams
                    WHERE user_id = ?
                )
                SELECT *
                FROM recent_activity
                CROSS JOIN goal_stats
                CROSS JOIN level_info
                CROSS JOIN mock_exam_stats
            ''', (self.user_id, self.user_id, self.user_id, self.user_id), fetch_all=False))

            stats = self._row_to_dict(stats_query)

            return {
                'study_streak': self._safe_get(stats, 'study_streak', 0),
                'total_hours': (self._safe_get(stats, 'total_minutes', 0) or 0) / 60,
                'avg_performance': self._safe_get(stats, 'avg_performance', 0),
                'questions_solved': (
                    self._safe_get(stats, 'total_correct', 0) +
                    self._safe_get(stats, 'total_incorrect', 0)
                ),
                'active_goals': self._safe_get(stats, 'active_goals', 0),
                'current_level': self._safe_get(stats, 'current_level', 1),
                'total_exams': self._safe_get(stats, 'total_exams', 0),
                'last_exam_date': self._safe_get(stats, 'last_exam')
            }
        except DatabaseError as e:
            st.error(f"Kullanıcı istatistikleri alınamadı: {str(e)}")
            return {
                'study_streak': 0,
                'total_hours': 0,
                'avg_performance': 0,
                'questions_solved': 0,
                'active_goals': 0,
                'current_level': 1,
                'total_exams': 0,
                'last_exam_date': None
            }

    def _show_motivation_message(self):
        """Display AI-generated motivation message"""
        st.markdown("### 💪 Motivasyon Mesajı")
        
        user_stats = self._get_user_stats()
        message = self.ai_service.generate_motivational_message(user_stats)
        
        with st.container():
            st.markdown(f"_{message}_")

    def _get_recent_achievements(self) -> List[Dict[str, Any]]:
        """Get user's recent achievements and badges"""
        try:
            achievements_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                SELECT 
                    a.title,
                    a.description,
                    a.date,
                    b.name as badge_name,
                    b.icon as badge_icon
                FROM achievements a
                LEFT JOIN user_badges ub ON a.user_id = ub.user_id AND ub.earned_date = a.date
                LEFT JOIN badges b ON ub.badge_id = b.id
                WHERE a.user_id = ?
                ORDER BY a.date DESC, ub.earned_date DESC
                LIMIT 5
            ''', (self.user_id,), fetch_all=True))
            
            return self._rows_to_dict_list(achievements_query)
            
        except DatabaseError as e:
            st.error(f"Başarı bilgileri alınamadı: {str(e)}")
            return []

    def _show_achievements(self):
        """Display recent achievements and badges"""
        st.markdown("### 🏆 Son Başarılar")
        
        achievements = self._get_recent_achievements()
        if achievements:
            for achievement in achievements:
                self._render_achievement_card(achievement)
        else:
            st.write("Henüz başarı kaydın bulunmuyor.")

    def _render_achievement_card(self, achievement: Dict[str, Any]):
        """Render individual achievement card"""
        with st.container():
            icon = self._safe_get(achievement, 'badge_icon', '🎯')
            title = self._safe_get(achievement, 'badge_name') or self._safe_get(achievement, 'title', '')
            date = datetime.strptime(
                self._safe_get(achievement, 'date', datetime.now().strftime('%Y-%m-%d')),
                '%Y-%m-%d'
            ).strftime('%d.%m.%Y')
            
            st.markdown(f"##### {icon} {title}")
            st.write(f"📅 {date}")
            if description := self._safe_get(achievement, 'description'):
                st.write(description)

    def _get_level_data(self) -> Dict[str, Any]:
        """Get user's level and XP data"""
        try:
            level_data_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT 
                    current_level,
                    current_xp,
                    total_xp
                FROM user_levels
                WHERE user_id = ?
            ''', (self.user_id,), fetch_all=False))

            level_data = self._row_to_dict(level_data_query)
            current_level = self._safe_get(level_data, 'current_level', 1)
            
            return {
                'current_level': current_level,
                'current_xp': self._safe_get(level_data, 'current_xp', 0),
                'total_xp': self._safe_get(level_data, 'total_xp', 0),
                'xp_for_next': self.gamification_service._calculate_xp_for_level(current_level + 1)
            }
        except DatabaseError as e:
            st.error(f"Seviye bilgileri alınamadı: {str(e)}")
            return {
                'current_level': 1,
                'current_xp': 0,
                'total_xp': 0,
                'xp_for_next': 1000
            }

    def _get_recent_xp_gains(self) -> List[Dict[str, Any]]:
        """Get recent XP gains"""
        try:
            gains_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                SELECT 
                    amount,
                    reason,
                    awarded_at
                FROM xp_bonuses
                WHERE user_id = ?
                ORDER BY awarded_at DESC
                LIMIT 3
            ''', (self.user_id,), fetch_all=True))
            
            return self._rows_to_dict_list(gains_query)
        except DatabaseError as e:
            st.error(f"XP kazanım bilgileri alınamadı: {str(e)}")
            return []

    def _show_level_progress(self):
        """Display user level and XP progress"""
        st.markdown("### 📈 Seviye İlerlemesi")
        
        level_data = self._get_level_data()
        
        st.metric("Seviye", level_data['current_level'])
        progress = level_data['current_xp'] / level_data['xp_for_next']
        st.progress(progress)
        st.write(f"XP: {level_data['current_xp']}/{level_data['xp_for_next']}")
        
        # Show recent XP gains
        recent_xp = self._get_recent_xp_gains()
        if recent_xp:
            st.markdown("#### Son Kazanılan XP")
            for xp in recent_xp:
                st.write(f"• +{self._safe_get(xp, 'amount', 0)} XP - {self._safe_get(xp, 'reason', '')}")

    def _show_quick_actions(self):
        """Display quick action buttons"""
        st.markdown("### ⚡ Hızlı İşlemler")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📝 Yeni Çalışma"):
                st.session_state.page = "study_calendar"
                st.rerun()
        
        with col2:
            if st.button("📊 Performans Raporu"):
                st.session_state.page = "performance"
                st.rerun()
        
        with col3:
            if st.button("✍️ Deneme Sonucu Gir"):
                st.session_state.page = "mock_exam"
                st.rerun()
        
        with col4:
            if st.button("🎯 Hedef Güncelle"):
                st.session_state.page = "goals"
                st.rerun()
    def _format_time_ago(self, date_str: str) -> str:
        """Format a date as a human-readable time ago string"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            now = datetime.now()
            diff = now - date

            if diff.days > 0:
                return f"{diff.days} gün önce"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                return f"{hours} saat önce"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                return f"{minutes} dakika önce"
            else:
                return "az önce"
        except (ValueError, TypeError):
            return "bilinmeyen tarih"

    def _show_error_message(self, error_message: str):
        """Display error message in a consistent format"""
        st.error(f"❌ {error_message}")

    def _show_success_message(self, success_message: str):
        """Display success message in a consistent format"""
        st.success(f"✅ {success_message}")

    def _validate_user_session(self) -> bool:
        """Validate user session"""
        if not self.user_id:
            self._show_error_message("Oturum bulunamadı. Lütfen tekrar giriş yapın.")
            return False
        return True

if __name__ == "__main__":
    if 'user_id' in st.session_state:
        home_page = HomePage(st.session_state.user_id)
        home_page.show()
    else:
        st.error("Lütfen giriş yapın.")

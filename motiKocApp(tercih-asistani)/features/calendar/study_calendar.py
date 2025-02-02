import streamlit as st
from datetime import datetime, timedelta, date
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, cast
import logging
import plotly.express as px
from core.database import DatabaseManager, DatabaseError
from services.gamification import GamificationService

# Configure logging
logger = logging.getLogger(__name__)

class StudyCalendar:
    def __init__(self, user_id: int):
        """Initialize StudyCalendar with user_id and database manager"""
        self.user_id = user_id
        self.db = DatabaseManager()
        self.gamification = GamificationService()

    def show(self):
        """Display the study calendar interface"""
        st.title("📅 Çalışma Takvimi")
        
        tab1, tab2, tab3 = st.tabs([
            "Takvim Görünümü",
            "Yeni Çalışma",
            "Raporlar"
        ])
        
        with tab1:
            self._show_calendar_view()
        with tab2:
            self._show_new_study_form()
        with tab3:
            self._show_study_reports()

    def _show_calendar_view(self):
        """Display calendar view of study sessions"""
        try:
            # Get current month's study data
            study_data = self._get_monthly_study_data()
            
            # Create calendar grid
            current_month = datetime.now()
            first_day = current_month.replace(day=1)
            last_day = (first_day + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            # Create week-based calendar
            weeks = self._create_calendar_weeks(first_day, last_day)
            
            # Display calendar
            st.markdown("### 📅 Aylık Görünüm")
            
            # Display weekday headers
            cols = st.columns(7)
            weekdays = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
            for i, col in enumerate(cols):
                col.markdown(f"**{weekdays[i]}**")
            
            # Display calendar grid
            self._display_calendar_grid(weeks, study_data)
            
        except Exception as e:
            logger.error(f"Error displaying calendar view: {e}")
            st.error("Takvim görünümü yüklenirken bir hata oluştu.")

    def _show_new_study_form(self):
        """Display form for adding new study sessions"""
        st.markdown("### 📝 Yeni Çalışma Ekle")
        
        try:
            with st.form("new_study_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    subject = st.selectbox(
                        "Ders",
                        self._get_subjects()
                    )
                    
                    topic = st.text_input("Konu")
                    study_date_input = st.date_input(
                        "Tarih",
                        value=datetime.now().date()
                    )
                    study_date = st.date_input(
                        "Tarih",
                        value=datetime.now().date()
                    )
                    study_date = cast(date, study_date_input)
                with col2:
                    duration = st.number_input(
                        "Süre (dakika)",
                        min_value=15,
                        max_value=180,
                        value=45,
                        step=15
                    )
                    
                    performance = st.slider(
                        "Verimlilik",
                        min_value=1,
                        max_value=5,
                        value=3
                    )
                    
                    notes = st.text_area("Notlar")
                
                submitted = st.form_submit_button("Kaydet")
                if submitted:
                    self._handle_study_form_submission(
                        subject, topic, study_date, duration, 
                        performance, notes
                    )
                    
        except Exception as e:
            logger.error(f"Error displaying study form: {e}")
            st.error("Çalışma formu yüklenirken bir hata oluştu.")

    def _show_study_reports(self):
        """Display study analytics and reports"""
        st.markdown("### 📊 Çalışma Raporları")
        
        try:
            # Time period selection
            period = st.selectbox(
                "Zaman Aralığı",
                ["Haftalık", "Aylık", "Yıllık"]
            )
            
            # Get study data for selected period
            study_data = self._get_study_data_by_period(period)
            
            if study_data:
                self._display_study_metrics(study_data)
                self._display_study_charts(study_data)
            else:
                st.info("Seçili dönem için çalışma verisi bulunamadı.")
                
        except Exception as e:
            logger.error(f"Error displaying study reports: {e}")
            st.error("Raporlar yüklenirken bir hata oluştu.")

    def _handle_study_form_submission(
        self, subject: str, topic: str, study_date: date, 
        duration: int, performance: int, notes: str
    ):
        """Handle the submission of a new study session"""
        try:
            date_str = study_date.strftime('%Y-%m-%d')
            
            success = self._add_study_session({
                'subject': subject,
                'topic': topic,
                'date': date_str,
                'duration': duration,
                'performance_rating': performance,
                'notes': notes
            })
            
            if success:
                st.success("Çalışma kaydedildi!")
                
                # Award XP
                xp_earned = self.gamification.calculate_xp_for_activity(
                    'study_session', 
                    duration=duration
                )
                if xp_earned > 0:
                    st.success(f"🎉 +{xp_earned} XP kazandın!")
                
                st.rerun()
            else:
                st.error("Çalışma kaydedilirken bir hata oluştu!")
                
        except Exception as e:
            logger.error(f"Error handling study form submission: {e}")
            st.error("Çalışma kaydedilirken bir hata oluştu!")

    def _create_calendar_weeks(
        self, first_day: datetime, last_day: datetime
    ) -> List[List[Optional[datetime]]]:
        """Create calendar weeks structure"""
        weeks = []
        current = first_day
        week = []
        
        # Add empty days if month doesn't start on Monday
        for _ in range(current.weekday()):
            week.append(None)
            
        while current <= last_day:
            if len(week) == 7:
                weeks.append(week)
                week = []
            week.append(current)
            current += timedelta(days=1)
            
        # Fill last week with None
        while len(week) < 7:
            week.append(None)
        weeks.append(week)
        
        return weeks

    def _display_calendar_grid(
        self, 
        weeks: List[List[Optional[datetime]]], 
        study_data: Dict[str, Dict[str, float]]
    ):
        """Display the calendar grid with study data"""
        for week in weeks:
            cols = st.columns(7)
            for day, col in zip(week, cols):
                if isinstance(day, datetime):
                    day_str = day.strftime('%Y-%m-%d')
                    day_data = study_data.get(day_str, {})
                    
                    if day_data:
                        col.markdown(
                            f"""
                            <div style='padding: 10px; 
                                      background-color: rgba(25, 135, 84, 0.1); 
                                      border-radius: 5px;'>
                                <strong>{day.day}</strong><br>
                                📚 {day_data['total_hours']:.1f} saat
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        col.markdown(f"{day.day}")
                else:
                    col.write("")

    def _display_study_metrics(self, study_data: Dict[str, Any]):
        """Display study metrics summary"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Toplam Çalışma",
                f"{study_data['total_hours']:.1f} saat"
            )
        
        with col2:
            st.metric(
                "Ortalama Günlük",
                f"{study_data['daily_average']:.1f} saat"
            )
        
        with col3:
            st.metric(
                "Verimlilik",
                f"{study_data['avg_performance']:.1f}/5"
            )

    def _display_study_charts(self, study_data: Dict[str, Any]):
        """Display study distribution charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Subject distribution
            df_subjects = pd.DataFrame(study_data['subject_distribution'])
            if not df_subjects.empty:
                fig_subjects = px.pie(
                    df_subjects,
                    values='hours',
                    names='subject',
                    title='Ders Dağılımı'
                )
                st.plotly_chart(fig_subjects, use_container_width=True)
        
        with col2:
            # Daily distribution
            df_daily = pd.DataFrame(study_data['daily_distribution'])
            if not df_daily.empty:
                fig_daily = px.bar(
                    df_daily,
                    x='date',
                    y='hours',
                    title='Günlük Çalışma'
                )
                st.plotly_chart(fig_daily, use_container_width=True)

    def _get_subjects(self) -> List[str]:
        """Get available subjects from database"""
        try:
            results = self.db.execute_query(
                'SELECT name FROM subjects',
                fetch_all=True
            )
            return [row['name'] for row in results] if results else []
        except DatabaseError as e:
            logger.error(f"Error fetching subjects: {e}")
            return []

    def _get_monthly_study_data(self) -> Dict[str, Any]:
        """Get study data for current month"""
        try:
            results = self.db.execute_query('''
                SELECT date, SUM(duration) as total_minutes
                FROM study_logs
                WHERE user_id = ?
                AND strftime('%Y-%m', date) = strftime('%Y-%m', 'now')
                GROUP BY date
            ''', (self.user_id,), fetch_all=True)
            
            return {
                row['date']: {
                    'total_hours': row['total_minutes'] / 60
                }
                for row in (results or [])
            }
        except DatabaseError as e:
            logger.error(f"Error fetching monthly data: {e}")
            return {}

    def _add_study_session(self, session_data: Dict[str, Any]) -> bool:
        """Add new study session to database"""
        try:
            # Insert study session
            session_id = self.db.insert_record('study_logs', {
                'user_id': self.user_id,
                'subject': session_data['subject'],
                'topic': session_data['topic'],
                'date': session_data['date'],
                'duration': session_data['duration'],
                'performance_rating': session_data['performance_rating'],
                'notes': session_data.get('notes', '')
            })
            
            # Award XP for study session
            xp = self.gamification.calculate_xp_for_activity(
                'study_session',
                duration=session_data['duration']
            )
            self.gamification.update_user_xp(self.user_id, xp)
            
            # Check and award badges
            self.gamification.check_and_award_badges(self.user_id)
            
            return True
            
        except DatabaseError as e:
            logger.error(f"Error adding study session: {e}")
            return False

    def _get_study_data_by_period(self, period: str) -> Optional[Dict[str, Any]]:
        """Get study data for selected time period"""
        try:
            # Set date filter based on period
            date_filter = {
                "Haftalık": "date >= date('now', '-7 days')",
                "Aylık": "date >= date('now', '-1 month')",
                "Yıllık": "date >= date('now', '-1 year')"
            }.get(period, "date >= date('now', '-7 days')")
            
            # Get summary data
            result = self.db.execute_query(f'''
                SELECT 
                    COUNT(DISTINCT date) as total_days,
                    SUM(duration) as total_minutes,
                    AVG(performance_rating) as avg_performance
                FROM study_logs
                WHERE user_id = ? AND {date_filter}
            ''', (self.user_id,))

            # Safely handle the result
            summary = cast(Dict[str, Any], result) if result else None
            if not summary or not summary.get('total_days'):
                return None
            
            # Get subject distribution
            subjects = self.db.execute_query(f'''
                SELECT subject, SUM(duration) as total_minutes
                FROM study_logs
                WHERE user_id = ? AND {date_filter}
                GROUP BY subject
            ''', (self.user_id,), fetch_all=True)
            
            # Get daily distribution
            daily = self.db.execute_query(f'''
                SELECT date, SUM(duration) as total_minutes
                FROM study_logs
                WHERE user_id = ? AND {date_filter}
                GROUP BY date
                ORDER BY date
            ''', (self.user_id,), fetch_all=True)
            
            # Safely access dictionary values
            total_minutes = summary.get('total_minutes', 0)
            total_days = summary.get('total_days', 1)  # Prevent division by zero
            avg_performance = summary.get('avg_performance', 0)
            
            return {
                'total_hours': total_minutes / 60 if total_minutes else 0,
                'daily_average': (total_minutes / 60) / total_days if total_minutes else 0,
                'avg_performance': avg_performance or 0,
                'subject_distribution': [
                    {
                        'subject': row['subject'],
                        'hours': row['total_minutes'] / 60
                    }
                    for row in (subjects or [])
                ],
                'daily_distribution': [
                    {
                        'date': row['date'],
                        'hours': row['total_minutes'] / 60
                    }
                    for row in (daily or [])
                ]
            }
            
        except DatabaseError as e:
            logger.error(f"Error fetching study data: {e}")
            return None


def show_study_calendar():
    """Initialize and display the StudyCalendar page"""
    user_id = st.session_state.get('user_id')
    if user_id is None:
        st.error("Kullanıcı kimliği bulunamadı.")
        return
    
    calendar = StudyCalendar(user_id=user_id)
    calendar.show()
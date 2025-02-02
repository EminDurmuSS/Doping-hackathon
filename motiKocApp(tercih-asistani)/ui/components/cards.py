# ui/components/cards.py
import streamlit as st
from typing import Optional, Dict, Any, List,Callable

class Card:
    @staticmethod
    def study_summary_card(data: Dict[str, Any]):
        """Display study summary card"""
        with st.container():
            st.markdown("""
            <div class="custom-card animate-fade-in">
                <h3>📚 Çalışma Özeti</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Toplam Çalışma", f"{data.get('total_hours', 0):.1f} saat")
            with col2:
                st.metric("Günlük Ortalama", f"{data.get('daily_average', 0):.1f} saat")
            with col3:
                st.metric("Çalışma Günü", str(data.get('study_days', 0)))

    @staticmethod
    def performance_card(data: Dict[str, Any]):
        """Display performance metrics card"""
        with st.container():
            st.markdown("""
            <div class="custom-card animate-fade-in">
                <h3>📊 Performans Metrikleri</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Ortalama Performans", 
                    f"{data.get('avg_performance', 0):.1f}/5"
                )
            with col2:
                st.metric(
                    "Başarı Oranı", 
                    f"{data.get('success_rate', 0):.1f}%"
                )

    @staticmethod
    def goal_card(goal_data: Dict[str, Any]):
        """Display goal tracking card"""
        with st.container():
            st.markdown("""
            <div class="custom-card animate-fade-in">
                <h3>🎯 Hedef Takibi</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(goal_data.get('progress', 0) / 100)
            st.write(f"Hedef: {goal_data.get('title', '')}")
            st.write(f"Son Tarih: {goal_data.get('deadline', '')}")

    @staticmethod
    def achievement_card(achievement: Dict[str, Any]):
        """Display achievement card"""
        with st.container():
            st.markdown(f"""
            <div class="custom-card animate-fade-in">
                <h4>{achievement.get('icon', '🏆')} {achievement.get('title', '')}</h4>
                <p>{achievement.get('description', '')}</p>
                <small>{achievement.get('date', '')}</small>
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def task_card(task: Dict[str, Any], on_complete: Callable[[Any], None]):
        """Display daily task card"""
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="custom-card">
                    <p>{task.get('description', '')}</p>
                    <small>💫 {task.get('xp_reward', 0)} XP</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if not task.get('completed', False):
                    if st.button("Tamamla", key=f"task_{task.get('id')}"):
                        on_complete(task.get('id'))

class AlertCard:
    @staticmethod
    def success(message: str):
        st.markdown(f"""
        <div class="alert-card success">
            <span>✅</span> {message}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def warning(message: str):
        st.markdown(f"""
        <div class="alert-card warning">
            <span>⚠️</span> {message}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def error(message: str):
        st.markdown(f"""
        <div class="alert-card error">
            <span>❌</span> {message}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def info(message: str):
        st.markdown(f"""
        <div class="alert-card info">
            <span>ℹ️</span> {message}
        </div>
        """, unsafe_allow_html=True)

class StatCard:
    @staticmethod
    def metric_card(title: str, value: Any, delta: Optional[Any] = None):
        """Display metric card with optional delta"""
        try:
            st.metric(
                label=title,
                value=value if value is not None else "0",
                delta=delta if delta is not None else None
            )
        except Exception:
            st.metric(
                label=title,
                value="0",
                delta=None
            )

    @staticmethod
    def progress_card(title: str, value: float, max_value: float):
        """Display progress card"""
        progress = value / max_value if max_value > 0 else 0
        st.markdown(f"### {title}")
        st.progress(progress)
        st.write(f"{value}/{max_value}")
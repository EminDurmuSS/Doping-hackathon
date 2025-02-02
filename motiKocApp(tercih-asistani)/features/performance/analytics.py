# features/performance/analytics.py
import streamlit as st
from typing import Dict, List, Any, Optional, Union, cast
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from services.ai_service import AIService
import json
import logging
from sqlite3 import Row

from core.database import DatabaseManager, DatabaseError, db_transaction

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db_manager = DatabaseManager()
        self.ai_service = AIService()

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

    def show(self):
        """Display performance analytics"""
        st.title("📊 Performans Analizi")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Genel Bakış",
            "Ders Analizi",
            "Deneme Sınavları",
            "AI Değerlendirme"
        ])

        with tab1:
            self._show_overview()
        with tab2:
            self._show_subject_analysis()
        with tab3:
            self._show_mock_exam_analysis()
        with tab4:
            self._show_ai_evaluation()

    def _show_overview(self):
        """Display general performance overview"""
        st.markdown("### 📈 Genel Performans")

        performance_data = self._get_performance_data()

        if not performance_data:
            st.error("Genel performans verisi alınamadı.")
            return

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Toplam Çalışma",
                f"{performance_data['total_hours']:.1f} saat",
                f"{performance_data['hours_change']:+.1f}%"
            )

        with col2:
            st.metric(
                "Ortalama Performans",
                f"{performance_data['avg_performance']:.1f}/5",
                f"{performance_data['performance_change']:+.1f}"
            )

        with col3:
            st.metric(
                "Çözülen Soru",
                performance_data['total_questions'],
                f"{performance_data['questions_change']:+d}"
            )

        with col4:
            st.metric(
                "Başarı Oranı",
                f"{performance_data['success_rate']:.1f}%",
                f"{performance_data['success_change']:+.1f}%"
            )

        # Display study trend
        st.markdown("#### 📊 Çalışma Trendi")
        fig = self._create_study_trend_chart(performance_data['daily_study'])
        st.plotly_chart(fig, use_container_width=True)

        # Display performance distribution
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Performans Dağılımı")
            fig = self._create_performance_dist_chart(performance_data['performance_dist'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### ⏰ Günlük Çalışma Dağılımı")
            fig = self._create_daily_dist_chart(performance_data['hourly_dist'])
            st.plotly_chart(fig, use_container_width=True)

    def _show_subject_analysis(self):
        """Display subject-wise performance analysis"""
        st.markdown("### 📚 Ders Analizi")

        subjects = self._get_subjects()
        if not subjects:
            st.info("Hiç ders bulunmuyor.")
            return

        subject = st.selectbox("Ders Seçin", subjects)
        subject_data = self._get_subject_performance(subject)

        if subject_data:
            # Display subject metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Toplam Çalışma",
                    f"{subject_data['total_hours']:.1f} saat"
                )

            with col2:
                st.metric(
                    "Ortalama Performans",
                    f"{subject_data['avg_performance']:.1f}/5"
                )

            with col3:
                st.metric(
                    "Başarı Oranı",
                    f"{subject_data['success_rate']:.1f}%"
                )

            # Display topic performance
            st.markdown("#### 📊 Konu Bazlı Performans")
            fig = self._create_topic_performance_chart(subject_data['topic_performance'])
            st.plotly_chart(fig, use_container_width=True)

            # Display improvement areas
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💪 Güçlü Konular")
                for topic in subject_data['strong_topics']:
                    st.markdown(f"- {topic['name']}: {topic['performance']:.1f}/5")

            with col2:
                st.markdown("#### 📈 Gelişim Alanları")
                for topic in subject_data['weak_topics']:
                    st.markdown(f"- {topic['name']}: {topic['performance']:.1f}/5")

            # Display recommendations
            st.markdown("#### 💡 Öneriler")
            for rec in subject_data['recommendations']:
                st.markdown(f"- {rec}")

        else:
            st.info(f"{subject} dersi için henüz veri bulunmuyor.")
    def _show_mock_exam_analysis(self):
        """Display mock exam performance analysis"""
        st.markdown("### 📝 Deneme Sınavı Analizi")

        mock_data = self._get_mock_exam_data()

        if mock_data.get('exams'):
            # Display overall progress
            st.markdown("#### 📈 Genel İlerleme")
            fig = self._create_mock_exam_trend_chart(mock_data['exam_trends'])
            st.plotly_chart(fig, use_container_width=True)

            # Display subject performance
            st.markdown("#### 📊 Ders Bazlı Performans")
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = self._create_subject_performance_chart(mock_data['subject_performance'])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 📈 Sıralama Değişimi")
                current_rank = mock_data.get('current_rank')
                prev_rank = mock_data.get('previous_rank')

                if current_rank is not None:
                    if prev_rank is not None:
                        change = prev_rank - current_rank
                        st.metric(
                            "Güncel Sıralama",
                            f"{current_rank:,}",
                            f"{change:+,}"
                        )
                    else:
                        st.metric("Güncel Sıralama", f"{current_rank:,}")
                else:
                    st.metric("Güncel Sıralama", "Veri Yok")

                st.markdown("#### 🎯 Hedef Sıralama")
                target_rank = self._get_target_rank()
                if target_rank:
                    if current_rank is not None:
                        diff = target_rank - current_rank
                        st.metric(
                            "Hedefe Uzaklık",
                            f"{abs(diff):,}",
                            "üstte" if diff < 0 else "geride"
                        )
                    else:
                        st.metric("Hedefe Uzaklık", "Veri Yok")
                else:
                    st.info("Hedef sıralama belirlenmemiş.")

            # Display detailed analysis
            st.markdown("#### 📊 Detaylı Analiz")
            tabs = st.tabs(["TYT", "AYT", "Soru Analizi"])

            with tabs[0]:
                self._show_tyt_analysis(mock_data.get('tyt_analysis'))
            with tabs[1]:
                self._show_ayt_analysis(mock_data.get('ayt_analysis'))
            with tabs[2]:
                self._show_question_analysis(mock_data.get('question_analysis'))
        else:
            st.info("Henüz deneme sınavı verisi bulunmuyor.")

    def _show_ai_evaluation(self):
        """Display AI-powered performance evaluation"""
        st.markdown("### 🤖 AI Değerlendirmesi")

        study_data = self._get_study_data()
        mock_data = self._get_mock_exam_data()
        target_data = self._get_target_data()

        analysis = self.ai_service.analyze_performance({
            'study_data': study_data,
            'mock_data': mock_data,
            'target_data': target_data
        })

        if analysis:
            # Display overall assessment
            st.markdown("#### 📊 Genel Değerlendirme")
            st.write(analysis.get('overall_assessment', 'Genel değerlendirme bulunamadı.'))

            # Display strengths and weaknesses
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 💪 Güçlü Yönler")
                for strength in analysis.get('strengths', []):
                    st.markdown(f"- {strength}")

            with col2:
                st.markdown("#### 📈 Gelişim Alanları")
                for area in analysis.get('improvement_areas', []):
                    st.markdown(f"- {area}")

            # Display predictions
            st.markdown("#### 🎯 Performans Tahminleri")
            predictions = analysis.get('predictions', {})
            col1, col2, col3 = st.columns(3)

            with col1:
                self._show_prediction_metric(
                    "Tahmini TYT Puanı",
                    predictions.get('tyt_score'),
                    predictions.get('tyt_change')
                )

            with col2:
                self._show_prediction_metric(
                    "Tahmini AYT Puanı",
                    predictions.get('ayt_score'),
                    predictions.get('ayt_change')
                )

            with col3:
                self._show_prediction_metric(
                    "Tahmini Sıralama",
                    predictions.get('rank'),
                    predictions.get('rank_change'),
                    is_rank=True
                )

            # Display recommendations
            st.markdown("#### 💡 Öneriler")
            tabs = st.tabs([
                "Çalışma Stratejisi",
                "Zaman Yönetimi",
                "Motivasyon"
            ])

            with tabs[0]:
                for rec in analysis.get('study_recommendations', []):
                    st.markdown(f"- {rec}")

            with tabs[1]:
                for rec in analysis.get('time_recommendations', []):
                    st.markdown(f"- {rec}")

            with tabs[2]:
                for rec in analysis.get('motivation_recommendations', []):
                    st.markdown(f"- {rec}")

            # Display action plan
            st.markdown("#### 📋 Aksiyon Planı")
            for i, action in enumerate(analysis.get('action_plan', []), 1):
                st.markdown(f"{i}. {action}")

        else:
            st.error("AI analizi oluşturulurken bir hata oluştu!")

    def _show_prediction_metric(self, label: str, value: Optional[float], 
                              change: Optional[float], is_rank: bool = False):
        """Helper method to display prediction metrics"""
        if value is not None and change is not None:
            if is_rank:
                st.metric(label, f"{value:,}", f"{change:+,}")
            else:
                st.metric(label, f"{value:.2f}", f"{change:+.2f}")
        else:
            st.metric(label, "Veri Yok")

    def _get_subjects(self) -> List[str]:
        """Get list of subjects"""
        try:
            subjects_query = cast(Optional[List[Row]], self.db_manager.execute_query(
                'SELECT name FROM subjects',
                tuple(),
                fetch_all=True
            ))
            return [row['name'] for row in (subjects_query or [])]
        except DatabaseError as e:
            logger.error(f"Error getting subjects: {str(e)}")
            return []

    def _get_performance_data(self) -> Optional[Dict[str, Any]]:
        """Get overall performance data"""
        try:
            with db_transaction() as conn:
                # Get current period stats
                current_query = cast(Optional[Row], self.db_manager.execute_query('''
                    SELECT 
                        COUNT(DISTINCT date) as study_days,
                        SUM(duration) as total_minutes,
                        AVG(performance_rating) as avg_performance,
                        COUNT(*) as total_questions
                    FROM study_logs
                    WHERE user_id = ?
                    AND date >= date('now', '-30 days')
                ''', (self.user_id,), fetch_all=False))

                current = self._row_to_dict(current_query)

                # Get previous period stats
                previous_query = cast(Optional[Row], self.db_manager.execute_query('''
                    SELECT 
                        COUNT(DISTINCT date) as study_days,
                        SUM(duration) as total_minutes,
                        AVG(performance_rating) as avg_performance,
                        COUNT(*) as total_questions
                    FROM study_logs
                    WHERE user_id = ?
                    AND date >= date('now', '-60 days')
                    AND date < date('now', '-30 days')
                ''', (self.user_id,), fetch_all=False))

                previous = self._row_to_dict(previous_query)

                # Calculate metrics
                total_hours = current.get('total_minutes', 0) / 60
                prev_hours = previous.get('total_minutes', 0) / 60
                hours_change = ((total_hours - prev_hours) / prev_hours * 100
                              if prev_hours > 0 else 0)
                performance_change = (
                    current.get('avg_performance', 0) - 
                    previous.get('avg_performance', 0)
                )

                questions_change = (
                    current.get('total_questions', 0) - 
                    previous.get('total_questions', 0)
                )
                # Get daily study data
                daily_study_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                    SELECT date, SUM(duration) / 60.0 as hours
                    FROM study_logs
                    WHERE user_id = ?
                    AND date >= date('now', '-30 days')
                    GROUP BY date
                    ORDER BY date
                ''', (self.user_id,), fetch_all=True))

                daily_study = self._rows_to_dict_list(daily_study_query)

                # Get performance distribution
                perf_dist_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                    SELECT 
                        performance_rating as rating,
                        COUNT(*) as count
                    FROM study_logs
                    WHERE user_id = ?
                    AND date >= date('now', '-30 days')
                    GROUP BY performance_rating
                    ORDER BY performance_rating
                ''', (self.user_id,), fetch_all=True))

                performance_dist = self._rows_to_dict_list(perf_dist_query)

                # Get hourly distribution
                hourly_dist_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                    SELECT 
                        strftime('%H', time) as hour,
                        SUM(duration) as minutes
                    FROM study_logs
                    WHERE user_id = ?
                    AND date >= date('now', '-30 days')
                    GROUP BY hour
                    ORDER BY hour
                ''', (self.user_id,), fetch_all=True))

                hourly_dist = self._rows_to_dict_list(hourly_dist_query)

                # Calculate success rate
                total_attempts = current.get('total_questions', 0)
                success_query = cast(Optional[Row], self.db_manager.execute_query('''
                    SELECT COUNT(*) as success_count
                    FROM study_logs
                    WHERE user_id = ?
                    AND date >= date('now', '-30 days')
                    AND performance_rating >= 4
                ''', (self.user_id,), fetch_all=False))

                successful_attempts = dict(success_query).get('success_count', 0) if success_query else 0

                success_rate = (successful_attempts / total_attempts * 100
                              if total_attempts > 0 else 0)

                return {
                    'total_hours': total_hours,
                    'hours_change': hours_change,
                    'avg_performance': current.get('avg_performance', 0),
                    'performance_change': performance_change,
                    'total_questions': current.get('total_questions', 0),
                    'questions_change': questions_change,
                    'study_days': current.get('study_days', 0),
                    'daily_study': daily_study,
                    'performance_dist': performance_dist,
                    'hourly_dist': hourly_dist,
                    'success_rate': success_rate,
                    'success_change': 0  # Placeholder
                }

        except DatabaseError as e:
            logger.error(f"Error getting performance data: {str(e)}")
            return None
    def _get_subject_performance(self, subject: str) -> Optional[Dict[str, Any]]:
        """Get subject performance data"""
        try:
            with db_transaction() as conn:
                # Get overall subject stats
                stats_query = cast(Optional[Row], self.db_manager.execute_query('''
                    SELECT 
                        SUM(duration) as total_minutes,
                        AVG(performance_rating) as avg_performance,
                        SUM(CASE WHEN performance_rating >= 4 THEN 1 ELSE 0 END) as success_count,
                        COUNT(*) as total_questions
                    FROM study_logs
                    WHERE user_id = ? AND subject = ?
                ''', (self.user_id, subject), fetch_all=False))

                stats = self._row_to_dict(stats_query)

                if not stats.get('total_minutes'):
                    return None

                # Calculate success rate
                success_rate = (
                    stats.get('success_count', 0) / stats.get('total_questions', 1) * 100
                    if stats.get('total_questions', 0) > 0 else 0
                )

                # Get topic performance
                topics_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                    SELECT 
                        topic,
                        AVG(performance_rating) as performance
                    FROM study_logs
                    WHERE user_id = ? AND subject = ?
                    GROUP BY topic
                    ORDER BY performance DESC
                ''', (self.user_id, subject), fetch_all=True))

                topics = self._rows_to_dict_list(topics_query)

                # Split into strong and weak topics
                strong_topics = []
                weak_topics = []

                for topic in topics:
                    topic_data = {
                        'name': topic.get('topic', ''),
                        'performance': topic.get('performance', 0)
                    }

                    if topic.get('performance', 0) >= 4:
                        strong_topics.append(topic_data)
                    elif topic.get('performance', 0) <= 3:
                        weak_topics.append(topic_data)

                # Get AI recommendations
                recommendations = self.ai_service.get_subject_recommendations(
                    subject,
                    {
                        'total_hours': stats.get('total_minutes', 0) / 60,
                        'avg_performance': stats.get('avg_performance', 0),
                        'topics': topics
                    }
                )

                return {
                    'total_hours': stats.get('total_minutes', 0) / 60,
                    'avg_performance': stats.get('avg_performance', 0),
                    'success_rate': success_rate,
                    'topic_performance': topics,
                    'strong_topics': strong_topics,
                    'weak_topics': weak_topics,
                    'recommendations': recommendations
                }

        except DatabaseError as e:
            logger.error(f"Error getting subject performance: {str(e)}")
            return None

    def _get_mock_exam_data(self) -> Dict[str, Any]:
        """Get mock exam data"""
        try:
            with db_transaction() as conn:
                # Get all mock exams
                exams_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                    SELECT *
                    FROM mock_exams
                    WHERE user_id = ?
                    ORDER BY exam_date DESC
                ''', (self.user_id,), fetch_all=True))

                exams = self._rows_to_dict_list(exams_query)

                if not exams:
                    return {'exams': []}

                # Get current and previous rank
                current_rank = exams[0].get('rank') if len(exams) >= 1 else None
                previous_rank = exams[1].get('rank') if len(exams) >= 2 else None

                # Process exam data
                exam_trends = []
                subject_performance = {}

                for exam in exams:
                    # Add to trends
                    exam_trends.append({
                        'date': exam.get('exam_date'),
                        'net': exam.get('total_net'),
                        'rank': exam.get('rank')
                    })

                    # Aggregate subject performance
                    results = json.loads(exam.get('subject_results', '{}'))
                    for subject, data in results.items():
                        if subject not in subject_performance:
                            subject_performance[subject] = {
                                'correct': 0,
                                'incorrect': 0,
                                'empty': 0
                            }

                        subject_performance[subject]['correct'] += data.get('correct', 0)
                        subject_performance[subject]['incorrect'] += data.get('incorrect', 0)
                        subject_performance[subject]['empty'] += data.get('empty', 0)

                # Convert subject performance to list
                subject_performance_list = [
                    {
                        'subject': subject,
                        **data
                    }
                    for subject, data in subject_performance.items()
                ]

                # Get TYT and AYT analysis
                tyt_analysis = self._analyze_exam_type(exams, 'TYT')
                ayt_analysis = self._analyze_exam_type(exams, 'AYT')

                # Get question analysis
                question_analysis = self._analyze_questions(exams)

                return {
                    'exams': exams,
                    'current_rank': current_rank,
                    'previous_rank': previous_rank,
                    'exam_trends': exam_trends,
                    'subject_performance': subject_performance_list,
                    'tyt_analysis': tyt_analysis,
                    'ayt_analysis': ayt_analysis,
                    'question_analysis': question_analysis
                }

        except DatabaseError as e:
            logger.error(f"Error getting mock exam data: {str(e)}")
            return {'exams': []}

    def _analyze_exam_type(self, exams: List[Dict[str, Any]], 
                         exam_type: str) -> Dict[str, Any]:
        """Analyze specific exam type performance"""
        type_exams = [
            exam for exam in exams 
            if exam.get('exam_type') == exam_type
        ]

        if not type_exams:
            return {}

        # Calculate trends
        trends = []
        for exam in type_exams:
            results = json.loads(exam.get('subject_results', '{}'))
            total_correct = sum(r.get('correct', 0) for r in results.values())
            total_incorrect = sum(r.get('incorrect', 0) for r in results.values())
            total_empty = sum(r.get('empty', 0) for r in results.values())

            trends.append({
                'date': exam.get('exam_date'),
                'correct': total_correct,
                'incorrect': total_incorrect,
                'empty': total_empty,
                'net': exam.get('total_net', 0)
            })

        # Calculate averages
        avg_net = sum(t['net'] for t in trends) / len(trends) if trends else 0
        avg_correct = sum(t['correct'] for t in trends) / len(trends) if trends else 0
        avg_incorrect = sum(t['incorrect'] for t in trends) / len(trends) if trends else 0
        avg_empty = sum(t['empty'] for t in trends) / len(trends) if trends else 0

        return {
            'trends': trends,
            'averages': {
                'net': avg_net,
                'correct': avg_correct,
                'incorrect': avg_incorrect,
                'empty': avg_empty
            }
        }

    def _analyze_questions(self, exams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze question performance"""
        all_results = []

        for exam in exams:
            results = json.loads(exam.get('subject_results', '{}'))
            for subject, data in results.items():
                all_results.append({
                    'date': exam.get('exam_date'),
                    'subject': subject,
                    'correct': data.get('correct', 0),
                    'incorrect': data.get('incorrect', 0),
                    'empty': data.get('empty', 0)
                })

        if not all_results:
            return {}

        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)

        # Calculate success rates
        success_rates = df.groupby('subject').apply(
            lambda x: {
                'correct_rate': (x['correct'].sum() / (x['correct'].sum() + x['incorrect'].sum()) * 100) 
                               if (x['correct'].sum() + x['incorrect'].sum()) > 0 else 0,
                'empty_rate': (x['empty'].sum() / len(x) * 100) 
                             if len(x) > 0 else 0,
                'total_questions': len(x)
            }
        ).to_dict()

        # Calculate trends
        trends = df.groupby(['date', 'subject']).agg({
            'correct': 'sum',
            'incorrect': 'sum',
            'empty': 'sum'
        }).reset_index()

        return {
            'success_rates': success_rates,
            'trends': trends.to_dict('records')
        }

    # Chart creation methods remain the same as they don't involve database operations
    def _create_study_trend_chart(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create study trend chart"""
        df = pd.DataFrame(data)
        fig = px.line(
            df,
            x='date',
            y='hours',
            title='Günlük Çalışma Trendi',
            labels={'hours': 'Saat', 'date': 'Tarih'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    def _create_performance_dist_chart(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create performance distribution chart"""
        df = pd.DataFrame(data)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['rating'],
            y=df['count'],
            marker_color='#1f77b4'
        ))
        fig.update_layout(
            title='Performans Dağılımı',
            xaxis_title='Performans Puanı',
            yaxis_title='Sayı',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    def _create_daily_dist_chart(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create daily distribution chart"""
        df = pd.DataFrame(data)
        fig = px.bar(
            df,
            x='hour',
            y='minutes',
            title='Saatlik Çalışma Dağılımı',
            labels={'minutes': 'Dakika', 'hour': 'Saat'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig
    def _create_topic_performance_chart(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create topic performance chart"""
        df = pd.DataFrame(data)
        fig = px.bar(
            df,
            x='topic',
            y='performance',
            title='Konu Bazlı Performans',
            color='performance',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    def _create_mock_exam_trend_chart(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create mock exam trend chart"""
        df = pd.DataFrame(data)
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['net'],
                name="Net",
                line=dict(color="#1f77b4")
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['rank'],
                name="Sıralama",
                line=dict(color="#ff7f0e")
            ),
            secondary_y=True
        )

        fig.update_layout(
            title='Deneme Sınavı Gelişimi',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        fig.update_yaxes(title_text="Net Sayısı", secondary_y=False)
        fig.update_yaxes(
            title_text="Sıralama",
            secondary_y=True,
            tickformat=",",
            autorange="reversed"
        )

        return fig

    def _create_subject_performance_chart(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create subject performance chart"""
        df = pd.DataFrame(data)
        fig = px.bar(
            df,
            x='subject',
            y=['correct', 'incorrect', 'empty'],
            title='Ders Bazlı Performans',
            labels={
                'value': 'Soru Sayısı',
                'subject': 'Ders',
                'variable': 'Durum'
            },
            color_discrete_map={
                'correct': '#2ecc71',
                'incorrect': '#e74c3c',
                'empty': '#95a5a6'
            }
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            barmode='stack'
        )
        return fig

    def _get_target_rank(self) -> Optional[int]:
        """Get user's target rank"""
        target_data = self._get_target_data()
        return target_data.get('target_rank')

    def _show_tyt_analysis(self, analysis: Optional[Dict[str, Any]]):
        """Display TYT analysis"""
        if not analysis:
            st.info("TYT analizi için yeterli veri bulunmuyor.")
            return

        # Display averages
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ortalama Net", f"{analysis['averages']['net']:.2f}")
        with col2:
            st.metric("Doğru Ortalaması", f"{analysis['averages']['correct']:.2f}")
        with col3:
            st.metric("Yanlış Ortalaması", f"{analysis['averages']['incorrect']:.2f}")
        with col4:
            st.metric("Boş Ortalaması", f"{analysis['averages']['empty']:.2f}")

        # Display trends
        fig = px.line(
            analysis['trends'],
            x='date',
            y=['correct', 'incorrect', 'empty', 'net'],
            title='TYT Gelişim Trendi',
            labels={
                'value': 'Soru Sayısı',
                'date': 'Tarih',
                'variable': 'Tip'
            }
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

    def _show_ayt_analysis(self, analysis: Optional[Dict[str, Any]]):
        """Display AYT analysis"""
        if not analysis:
            st.info("AYT analizi için yeterli veri bulunmuyor.")
            return

        # Display averages
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ortalama Net", f"{analysis['averages']['net']:.2f}")
        with col2:
            st.metric("Doğru Ortalaması", f"{analysis['averages']['correct']:.2f}")
        with col3:
            st.metric("Yanlış Ortalaması", f"{analysis['averages']['incorrect']:.2f}")
        with col4:
            st.metric("Boş Ortalaması", f"{analysis['averages']['empty']:.2f}")

        # Display trends
        fig = px.line(
            analysis['trends'],
            x='date',
            y=['correct', 'incorrect', 'empty', 'net'],
            title='AYT Gelişim Trendi',
            labels={
                'value': 'Soru Sayısı',
                'date': 'Tarih',
                'variable': 'Tip'
            }
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

    def _show_question_analysis(self, analysis: Optional[Dict[str, Any]]):
        """Display question analysis"""
        if not analysis:
            st.info("Soru analizi için yeterli veri bulunmuyor.")
            return

        # Display success rates
        st.markdown("#### 📊 Başarı Oranları")
        for subject, rates in analysis.get('success_rates', {}).items():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{subject} Doğru Oranı", f"{rates['correct_rate']:.1f}%")
            with col2:
                st.metric("Boş Oranı", f"{rates['empty_rate']:.1f}%")
            with col3:
                st.metric("Toplam Soru", rates['total_questions'])

        # Display trends
        st.markdown("#### 📈 Soru Çözüm Trendi")
        trends = analysis.get('trends', [])
        if trends:
            df = pd.DataFrame(trends)
            fig = px.line(
                df,
                x='date',
                y=['correct', 'incorrect', 'empty'],
                color='subject',
                title='Ders Bazlı Soru Çözüm Trendi',
                labels={
                    'value': 'Soru Sayısı',
                    'date': 'Tarih',
                    'subject': 'Ders'
                }
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Soru çözüm trendi için yeterli veri bulunmuyor.")

    def _get_study_data(self) -> Dict[str, Any]:
        """Get user's study data for AI analysis"""
        try:
            study_logs_query = cast(Optional[List[Row]], self.db_manager.execute_query('''
                SELECT 
                    date,
                    duration,
                    performance_rating,
                    subject,
                    topic
                FROM study_logs
                WHERE user_id = ?
                ORDER BY date DESC
            ''', (self.user_id,), fetch_all=True))

            study_logs = self._rows_to_dict_list(study_logs_query)

            return {
                'study_logs': [
                    {
                        'date': log.get('date'),
                        'duration': log.get('duration'),
                        'performance_rating': log.get('performance_rating'),
                        'subject': log.get('subject'),
                        'topic': log.get('topic')
                    }
                    for log in study_logs
                ]
            }
        except DatabaseError as e:
            logger.error(f"Error getting study data: {str(e)}")
            return {'study_logs': []}

    def _get_target_data(self) -> Dict[str, Any]:
        """Get user's target data"""
        try:
            target_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT target_university, target_department, target_rank
                FROM users
                WHERE id = ?
            ''', (self.user_id,), fetch_all=False))

            return self._row_to_dict(target_query)
        except DatabaseError as e:
            logger.error(f"Error getting target data: {str(e)}")
            return {}

def show_performance_analytics():
    """Main function to show performance analytics"""
    user_id = st.session_state.get('user_id')
    if user_id:
        analytics = PerformanceAnalytics(user_id)
        analytics.show()
    else:
        st.error("Lütfen giriş yapın.")


if __name__ == "__main__":
    show_performance_analytics()        
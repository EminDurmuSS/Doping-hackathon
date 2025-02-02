# services/chart_service.py
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import pandas as pd

class ChartService:
    @staticmethod
    def create_study_time_chart(data: List[Dict[str, Any]]) -> go.Figure:
        """Create study time distribution chart"""
        df = pd.DataFrame(data)
        fig = px.bar(
            df,
            x='date',
            y='duration',
            color='subject',
            title='Günlük Çalışma Süreleri',
            labels={'duration': 'Süre (dakika)', 'date': 'Tarih'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_performance_radar(data: Dict[str, float]) -> go.Figure:
        """Create performance radar chart"""
        categories = list(data.keys())
        values = list(data.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Performans'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_progress_chart(data: List[Dict[str, Any]]) -> go.Figure:
        """Create progress tracking chart"""
        df = pd.DataFrame(data)
        fig = px.line(
            df,
            x='date',
            y='score',
            color='subject',
            title='Konu Bazlı İlerleme',
            labels={'score': 'Başarı Puanı', 'date': 'Tarih'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_mock_exam_comparison(data: List[Dict[str, Any]]) -> go.Figure:
        """Create mock exam comparison chart"""
        df = pd.DataFrame(data)
        fig = go.Figure()
        
        # Add traces for each subject
        for subject in df['subject'].unique():
            subject_data = df[df['subject'] == subject]
            fig.add_trace(go.Scatter(
                x=subject_data['date'],
                y=subject_data['net'],
                name=subject,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title='Deneme Sınavı Gelişimi',
            xaxis_title='Tarih',
            yaxis_title='Net Sayısı',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_subject_distribution_pie(data: List[Dict[str, Any]]) -> go.Figure:
        """Create subject distribution pie chart"""
        df = pd.DataFrame(data)
        fig = px.pie(
            df,
            values='duration',
            names='subject',
            title='Ders Dağılımı'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_weekly_activity_heatmap(data: List[Dict[str, Any]]) -> go.Figure:
        """Create weekly activity heatmap"""
        df = pd.DataFrame(data)
        fig = px.density_heatmap(
            df,
            x='weekday',
            y='hour',
            z='duration',
            title='Haftalık Aktivite Haritası',
            labels={'duration': 'Çalışma Süresi (dk)'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig
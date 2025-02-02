# features/voice/guidance.py

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import time

from services.ai_service import AIService  # Import AIService

# Initialize AIService
ai_service = AIService()

def show_voice_guidance():
    st.title("🎧 Sesli Rehberlik")
    
    guidance_type = st.selectbox(
        "Rehberlik Türü",
        ["Meditasyon", "Nefes Egzersizi", "Motivasyon", "Konu Anlatımı"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if guidance_type == "Meditasyon":
            st.markdown("""
            <div class="custom-card">
                <h3>Meditasyon Seansı</h3>
                <p>Sınav kaygısını azaltmak için özel olarak tasarlanmış meditasyon seansları.</p>
            </div>
            """, unsafe_allow_html=True)
            
            meditation_type = st.selectbox(
                "Meditasyon Türü",
                ["Sınav Kaygısı Azaltma (10 dk)",
                 "Odaklanma Meditasyonu (5 dk)",
                 "Rahatlama Seansı (15 dk)"]
            )
            
            if st.button("Seansı Başlat"):
                duration = int(meditation_type.split("(")[1].split(" dk")[0]) * 60
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                meditation_prompts = [
                    "Derin bir nefes al...",
                    "Zihnini sakinleştir...",
                    "Kendini güvende hisset...",
                    "Stresini bırak...",
                    "Başaracağına inan..."
                ]
                
                for i in range(duration):
                    progress = (i + 1) / duration
                    progress_bar.progress(progress)
                    
                    if i % 12 == 0:
                        status_text.write(
                            meditation_prompts[int(i/12) % len(meditation_prompts)]
                        )
                    time.sleep(1)
                
                st.success("Meditasyon tamamlandı! 🧘‍♂️")
                
                # Update meditation count
                if "meditation_count" not in st.session_state:
                    st.session_state.meditation_count = 0
                st.session_state.meditation_count += 1
        elif guidance_type == "Nefes Egzersizi":
            st.markdown("""
            <div class="custom-card">
                <h3>Nefes Egzersizleri</h3>
                <p>Sınav öncesi ve çalışma aralarında rahatlamanıza yardımcı olacak nefes teknikleri.</p>
            </div>
            """, unsafe_allow_html=True)
            
            breathing_type = st.selectbox(
                "Nefes Tekniği",
                ["4-7-8 Tekniği", "Kutu Nefes", "Sakinleştirici Nefes"]
            )
            
            if st.button("Egzersizi Başlat"):
                cycles = 5
                if breathing_type == "4-7-8 Tekniği":
                    steps = [
                        ("Nefes Al", 4),
                        ("Tut", 7),
                        ("Nefes Ver", 8)
                    ]
                else:  # Kutu Nefes veya Sakinleştirici Nefes
                    steps = [
                        ("Nefes Al", 4),
                        ("Tut", 4),
                        ("Nefes Ver", 4),
                        ("Bekle", 4)
                    ]
                
                progress_bar = st.progress(0)
                instruction_text = st.empty()
                
                total_duration = sum(step[1] for step in steps) * cycles
                current_time = 0
                
                for cycle in range(cycles):
                    for instruction, duration in steps:
                        instruction_text.markdown(f"""
                        <div style='text-align: center; font-size: 24px;'>
                            {instruction}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for _ in range(duration):
                            current_time += 1
                            progress = current_time / total_duration
                            progress_bar.progress(progress)
                            time.sleep(1)
                
                st.success("Egzersiz tamamlandı! 🌬️")
    
        elif guidance_type == "Motivasyon":
            st.markdown("""
            <div class="custom-card">
                <h3>Günlük Motivasyon</h3>
                <p>YKS hazırlık sürecinde size güç verecek motivasyon mesajları.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Fetch user data for personalized motivation
            conn = sqlite3.connect('motikoc.db')
            user_data = pd.read_sql_query('''
            SELECT name, grade, target_university, target_department, target_rank
            FROM users WHERE id = ?
        ''', conn, params=(st.session_state.user_id,))
    
            conn.close()
            
            days_to_exam = (datetime(2025, 6, 15) - datetime.now()).days
            
            motivation_speech = ai_service.generate_motivational_message({
                'name': user_data.iloc[0]['name'],
                'grade': user_data.iloc[0]['grade'],
                'target_university': user_data.iloc[0]['target_university'],
                'target_department': user_data.iloc[0]['target_department'],
                'days_to_exam': days_to_exam
            })
            
            st.markdown(f"""
            <div class="custom-card" style="background-color: #2d3748; padding: 20px; border-radius: 10px;">
                <h4 style="color: #f8f9fa;">💪 Motivasyon Konuşması</h4>
                <p style="color: #f8f9fa; font-size: 18px;">{motivation_speech}</p>
            </div>
            """, unsafe_allow_html=True)
    
        elif guidance_type == "Konu Anlatımı":
            st.markdown("""
            <div class="custom-card">
                <h3>Sesli Konu Anlatımı</h3>
                <p>Önemli konuların özet anlatımları ve tekrar rehberi.</p>
            </div>
            """, unsafe_allow_html=True)
            
            subject = st.selectbox(
                "Ders Seçin",
                ["Matematik", "Fizik", "Kimya", "Biyoloji", "Türkçe", "Tarih"]
            )
            
            topics = {
                "Matematik": ["Türev", "İntegral", "Limit", "Fonksiyonlar"],
                "Fizik": ["Kuvvet", "Hareket", "Elektrik", "Manyetizma"],
                "Kimya": ["Asitler", "Gazlar", "Karışımlar", "Tepkimeler"],
                "Biyoloji": ["Hücre", "Sistemler", "Kalıtım", "Evrim"],
                "Türkçe": ["Paragraf", "Dil Bilgisi", "Anlatım", "Şiir"],
                "Tarih": ["Osmanlı", "Cumhuriyet", "İnkılaplar", "Çağdaş"]
            }
            
            topic = st.selectbox("Konu Seçin", topics[subject])
            
            if st.button("Anlatımı Başlat"):
                # Replace the following with the actual question related to the topic
                question = f"{subject} dersinin {topic} konusuyla ilgili örnek bir soru."
                
                explanation = ai_service.explain_solution(
                    question=question,
                    subject=subject
                )
                
                st.markdown(f"""
                <div class="custom-card">
                    <h4>📚 {subject} - {topic}</h4>
                    <p>{explanation.get('explanation', 'Açıklama bulunamadı.')}</p>
                </div>
                """, unsafe_allow_html=True)

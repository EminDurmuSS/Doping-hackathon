# university/recommender.py

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import google.generativeai as genai
from dataclasses import dataclass, field
import re
from enum import Enum, auto
import json
import os
import asyncio
import streamlit as st
import plotly.express as px
from datetime import datetime
import nest_asyncio
import logging

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Intent Types
class IntentType(Enum):
    PROGRAM = auto()
    UNIVERSITY = auto()
    CITY = auto()
    SCORE_TYPE = auto()
    RANKING = auto()
    SCHOLARSHIP = auto()
    COMPARISON = auto()
    FACULTY = auto()
    FEES = auto()
    QUOTA = auto()

# Define Filter Criteria using dataclass
@dataclass
class FilterCriteria:
    min_ranking: Optional[int] = None
    max_ranking: Optional[int] = None
    universities: List[str] = field(default_factory=list)
    programs: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    score_types: List[str] = field(default_factory=list)
    scholarship_percentage: Optional[int] = None
    faculty_types: List[str] = field(default_factory=list)
    university_types: List[str] = field(default_factory=list)
    max_fee: Optional[float] = None
    min_quota: Optional[int] = None
    language_types: List[str] = field(default_factory=list)
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    
    def __post_init__(self):
        self.universities = self.universities or []
        self.programs = self.programs or []
        self.cities = self.cities or []
        self.score_types = self.score_types or []
        self.faculty_types = self.faculty_types or []
        self.university_types = self.university_types or []
        self.language_types = self.language_types or []

class UniversityRecommender:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro-002"):
        """Initialize the recommender with Gemini API key and model."""
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError("Gemini API key must be provided or set as an environment variable.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.system_prompt = """
Sen bir üniversite öneri sistemi asistanısın. Kullanıcının verdiği soruyu analiz et ve kullanıcının tercih kriterlerini ayrıştırarak aşağıdaki bilgileri çıkar. 
Kullanıcının ifadesini en yalın haliyle anla. Örneğin, "yazılımcılık" ifadesi geldiğinde bunu "yazılım mühendisi" olarak algıla.

Özellikle "fakülte" ile ilgili tercihleri "fakulte_tercihleri" altında, spesifik program tercihlerini ise "tercih_edilen_programlar_bolumler" alanında 
tanımla. 

Kullanıcı "mühendislik", "hukuk", "tıp" gibi fakülte isimlerini belirttiğinde, bunu "fakulte_tercihleri" altında değerlendir. Ancak, 
spesifik bir program adı verdiğinde (örneğin "bilgisayar mühendisliği"), bunu "tercih_edilen_programlar_bolumler" altında değerlendir. Bu ayrımı 
sağlamak için aşağıdaki şablonu takip et.

Çıkarılacak Bilgiler:
- tercih_edilen_universiteler: Örneğin, "İstanbul Üniversitesi", "Boğaziçi Üniversitesi".
- tercih_edilen_programlar_bolumler: Örneğin, "bilgisayar mühendisliği", "endüstri mühendisliği".
- tercih_edilen_sehirler: Örneğin, "İstanbul", "Ankara".
- taban_puan_araligi_veya_belirli_bir_puan_degeri: Puan değeri veya aralık (örneğin 450.0 veya 400.0-500.0).
- basari_sirasi_araligi_veya_belirli_bir_siralama: Belirli bir başarı sırası veya aralık (örneğin 5000 veya 3000-10000).
- burs_gereksinimleri: Yüzde olarak burs gereksinimi (örneğin %50 veya %100).
- fakulte_tercihleri: Örneğin, "Mühendislik Fakültesi", "Hukuk Fakültesi".
- universite_turu: Üniversite türü (örneğin, "devlet", "vakıf", "kıbrıs").
- puan türü: SAY, EA, SÖZ, DİL, TYT gibi. Kullanıcı "sayısal" derse SAY, "eşit ağırlık" derse EA, "sözel" derse SÖZ, "dil" derse DİL olarak döndür. 
  2 yıllık veya TYT puan türü belirtilmişse, TYT olarak döndür.
- maksimum_ucret: Kullanıcının belirttiği maksimum ücret (örneğin, 50000.0).
- dil_tercihi: Program dili (örneğin, "İngilizce", "Fransızca", "Almanca").
- minimum_kontenjan_gereksinimleri: En az kontenjan değeri (örneğin, 10).

**Dikkat Edilmesi Gerekenler**:
- Çıktıyı JSON formatında döndür ve tüm alanları ekle. Eğer bilgi verilmediyse, ilgili alanı boş string ("") olarak döndür.
- JSON formatı şu şekilde olmalıdır:
  {
      "tercih_edilen_universiteler": [],
      "tercih_edilen_programlar_bolumler": [],
      "tercih_edilen_sehirler": [],
      "taban_puan_araligi_veya_belirli_bir_puan_degeri": "",
      "basari_sirasi_araligi_veya_belirli_bir_siralama": "",
      "burs_gereksinimleri": "",
      "fakulte_tercihleri": [],
      "universite_turu": "",
      "puan_turu": "",
      "maksimum_ucret": "",
      "dil_tercihi": [],
      "minimum_kontenjan_gereksinimleri": ""
  }

**Örnek Kullanıcı Soruları**:
- "İstanbul'da mühendislik fakültesinde %50 burslu bilgisayar mühendisliği istiyorum."
- "Yazılımcılık bölümü istiyorum." (Bu durumda "yazılım mühendisi" olarak algılanmalı.)
- "Tıp fakültesi için başarı sıralamam 5000'den iyi olan programlar."
- "Eşit ağırlık ile %100 burslu, 400-500 puan aralığında hukuk bölümleri."

Kullanıcının sorusunu analiz ederek, bu bilgileri doğru JSON formatında ver.
    """

    def _analyze_intent(self, question: str) -> Dict[str, Any]:
        """
        Analyze user question using LLM to detect intents and filtering criteria.

        Args:
            question (str): The user's natural language query.

        Returns:
            Dict[str, Any]: Parsed intent data in JSON format.
        """
        try:
            prompt = f"{self.system_prompt}\n\nKullanıcı sorusu: {question}"
            response = self.model.generate_content(prompt)
            
            # Log the raw response for debugging
            response_text = response.text.strip() if response.text else "(No content)"
            logger.info(f"Raw response: {response_text}")
            
            # Clean the response by removing backticks and unnecessary formatting
            cleaned_response = re.sub(r'```json', '', response_text, flags=re.IGNORECASE)
            cleaned_response = re.sub(r'```', '', cleaned_response)
            cleaned_response = cleaned_response.strip()
            logger.info(f"Cleaned response: {cleaned_response}")  # For debugging

            # Parse JSON
            intent_data = json.loads(cleaned_response)
            return intent_data
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {response_text}")
            return {}
        except Exception as e:
            logger.error(f"Error in intent analysis: {e}")
            return {}

    def _create_filter_criteria(self, intent_data: Dict[str, Any]) -> FilterCriteria:
        """
        Convert LLM intent analysis into structured filter criteria.

        Args:
            intent_data (Dict[str, Any]): The parsed intent data.

        Returns:
            FilterCriteria: An instance containing the filter criteria.
        """
        criteria = FilterCriteria()
        
        criteria.programs = intent_data.get('tercih_edilen_programlar_bolumler', [])
        criteria.cities = intent_data.get('tercih_edilen_sehirler', [])
        puan_turu = intent_data.get('puan_turu', '')
        if puan_turu:
            puan_map = {
                "sayısal": "SAY",
                "eşit ağırlık": "EA",
                "sözel": "SÖZ",
                "dil": "DİL",
                "2 yıllık": "TYT",
                "TYT": "TYT"
            }
            criteria.score_types = [puan_map.get(puan_turu.lower(), puan_turu.upper())]
        
        basari_sirasi = intent_data.get('basari_sirasi_araligi_veya_belirli_bir_siralama', '')

        # Split ranking range if exists
        if '-' in basari_sirasi:
            try:
                min_ranking, max_ranking = map(int, basari_sirasi.split('-'))
                criteria.min_ranking = min_ranking
                criteria.max_ranking = max_ranking
            except ValueError:
                logger.warning(f"Geçersiz aralık formatı: {basari_sirasi}")
        elif basari_sirasi:
            try:
                ranking = int(basari_sirasi)
                criteria.min_ranking = ranking
                criteria.max_ranking = ranking
            except ValueError:
                logger.warning(f"Geçersiz sıralama formatı: {basari_sirasi}")

        # Taban Puan
        taban_puan = intent_data.get('taban_puan_araligi_veya_belirli_bir_puan_degeri', '')
        
        if isinstance(taban_puan, str) and '-' in taban_puan:
            try:
                min_score, max_score = map(float, taban_puan.split('-'))
                criteria.min_score = min_score
                criteria.max_score = max_score
            except ValueError:
                logger.warning(f"Geçersiz puan aralığı formatı: {taban_puan}")
        elif isinstance(taban_puan, (float, int)):  # Single score
            criteria.max_score = float(taban_puan)

        # Scholarship Percentage
        burs = intent_data.get('burs_gereksinimleri', '')
        if burs:
            match = re.search(r'(\d+)', burs)
            if match:
                try:
                    burs_percentage = int(match.group(1))
                    criteria.scholarship_percentage = burs_percentage
                except ValueError:
                    logger.warning(f"Geçersiz burs formatı: {burs}")
            else:
                logger.warning(f"Burs yüzdesi bulunamadı: {burs}")

        # Faculty Types
        criteria.faculty_types = intent_data.get('fakulte_tercihleri', [])

        # University Types
        universite_turu = intent_data.get('universite_turu', '')
        if universite_turu:
            criteria.university_types = [universite_turu.lower()]
        
        # Maximum Fee
        maks_ucret = intent_data.get('maksimum_ucret', '')
        if maks_ucret:
            try:
                criteria.max_fee = float(maks_ucret)
            except ValueError:
                logger.warning(f"Geçersiz maksimum ücret formatı: {maks_ucret}")

        # Language Preference
        criteria.language_types = intent_data.get('dil_tercihi', [])

        # Minimum Quota
        min_quota = intent_data.get('minimum_kontenjan_gereksinimleri', '')
        if min_quota:
            try:
                criteria.min_quota = int(min_quota)
            except ValueError:
                logger.warning(f"Geçersiz kontenjan formatı: {min_quota}")

        return criteria

    def _apply_filters(self, df: pd.DataFrame, criteria: FilterCriteria) -> pd.DataFrame:
        """
        Apply filtering criteria to the dataframe.

        Args:
            df (pd.DataFrame): The original university programs data.
            criteria (FilterCriteria): The selected filter criteria.

        Returns:
            pd.DataFrame: The filtered university programs data.
        """
        filtered_df = df.copy()
        
        try:
            # Ranking Filters
            if criteria.min_ranking is not None:
                filtered_df = filtered_df[filtered_df['Başarı Sırası 2023'] >= criteria.min_ranking]
            if criteria.max_ranking is not None:
                filtered_df = filtered_df[filtered_df['Başarı Sırası 2023'] <= criteria.max_ranking]

            # Universities Filter
            if criteria.universities:
                pattern = '|'.join(map(re.escape, criteria.universities))
                filtered_df = filtered_df[filtered_df['Üniversite'].str.contains(pattern, case=False, na=False)]

            # Programs Filter
            if criteria.programs:
                pattern = '|'.join(map(re.escape, criteria.programs))
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(pattern, case=False, na=False)]
                logger.info(f"After filtering programs: {len(filtered_df)}")

            # Cities Filter
            if criteria.cities:
                pattern = '|'.join(map(re.escape, criteria.cities))
                filtered_df = filtered_df[filtered_df['Şehir'].str.contains(pattern, case=False, na=False)]

            # Score Types Filter
            if criteria.score_types:
                pattern = '|'.join(map(re.escape, criteria.score_types))
                filtered_df = filtered_df[filtered_df['Puan Türü'].str.contains(pattern, case=False, na=False)]

            # Language Types Filter
            if criteria.language_types:
                language_pattern = '|'.join(map(re.escape, criteria.language_types))
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(language_pattern, case=False, na=False)]

            # Scholarship Percentage Filter
            if criteria.scholarship_percentage is not None:
                scholarship_str = f"%{criteria.scholarship_percentage}"
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(scholarship_str, case=False, na=False)]

            # Faculty Types Filter
            if criteria.faculty_types:
                pattern = '|'.join(map(re.escape, criteria.faculty_types))
                filtered_df = filtered_df[filtered_df['Fakülte'].str.contains(pattern, case=False, na=False)]

            # University Types Filter
            if criteria.university_types:
                pattern = '|'.join(map(re.escape, criteria.university_types))
                filtered_df = filtered_df[filtered_df['Üni.Türü'].str.contains(pattern, case=False, na=False)]

            # Maximum Fee Filter
            if criteria.max_fee is not None:
                filtered_df['Ücret (KDV Hariç)'] = pd.to_numeric(filtered_df['Ücret (KDV Hariç)'], errors='coerce').fillna(0)
                filtered_df = filtered_df[filtered_df['Ücret (KDV Hariç)'] <= criteria.max_fee]

            # Minimum Quota Filter
            if criteria.min_quota is not None:
                filtered_df['Kontenjan 2023'] = pd.to_numeric(filtered_df['Kontenjan 2023'], errors='coerce').fillna(0)
                filtered_df = filtered_df[filtered_df['Kontenjan 2023'] >= criteria.min_quota]

            # Taban Puan Filters
            if criteria.min_score is not None:
                filtered_df = filtered_df[filtered_df['Taban Puan 2023'] >= criteria.min_score]
            if criteria.max_score is not None:
                filtered_df = filtered_df[filtered_df['Taban Puan 2023'] <= criteria.max_score]

            # Sorting
            filtered_df = filtered_df.sort_values('Başarı Sırası 2023', ascending=True)
            return filtered_df

        except Exception as e:
            logger.error(f"Error in applying filters: {e}")
            return df

    def _generate_response(self, filtered_df: pd.DataFrame, question: str) -> str:
        """
        Generate a natural language response based on the filtered results.

        Args:
            filtered_df (pd.DataFrame): The filtered university programs data.
            question (str): The user's original question.

        Returns:
            str: The AI-generated response.
        """
        try:
            if filtered_df.empty:
                return "Üzgünüm, arama kriterlerinize uygun bir program bulamadım."

            results_summary = filtered_df.head(20).to_dict('records')

            prompt = f"""
Kullanıcının sorusu: {question}

En iyi 5 sonuç:
{json.dumps(results_summary, ensure_ascii=False, indent=2)}

Lütfen bu sonuçları analiz ederek Türkçe olarak detaylı ve yardımcı bir cevap ver:
1. Ana bulguları özetle.
2. Önemli eğilimleri veya kalıpları vurgula.
3. Spesifik önerilerde bulun.
4. Önemli dikkat edilmesi gereken noktaları belirt.
5. Öğrenciye sonraki adımlar için tavsiyeler ver.

Cevabı doğal ve samimi bir üslupla yaz, ancak kısa ve odaklı tut.
"""

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error in generating response: {e}")
            return "Sonuçları değerlendirirken bir hata oluştu. Lütfen daha sonra tekrar deneyin."

    def process_question(self, question: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Process user question and return filtered results with explanation.

        Args:
            question (str): The user's natural language query.
            df (pd.DataFrame): The original university programs data.

        Returns:
            Tuple[pd.DataFrame, str]: The filtered data and the AI-generated response.
        """
        try:
            intent_data = self._analyze_intent(question)
            if not intent_data:
                return df, "Sorunuzu anlamakta zorlanıyorum. Lütfen daha net bir ifade kullanın."

            criteria = self._create_filter_criteria(intent_data)
            filtered_df = self._apply_filters(df, criteria)
            response = self._generate_response(filtered_df, question)
            return filtered_df, response
                
        except Exception as e:
            logger.error(f"Error in processing question: {e}")
            return df, "İşlem sırasında bir hata oluştu. Lütfen daha sonra tekrar deneyin."

class UniversityRecommenderInterface:
    def __init__(self, api_key: str):
        """Initialize the interface with necessary configurations and session state."""
        self._initialize_session_state()
        self.recommender = self._initialize_recommender(api_key)
        self.df = self._load_data()

    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'filtered_results' not in st.session_state:
            st.session_state.filtered_results = None
        if 'current_response' not in st.session_state:
            st.session_state.current_response = None

    def _initialize_recommender(self, api_key: str) -> UniversityRecommender:
        """Initialize the UniversityRecommender with the provided API key."""
        return UniversityRecommender(api_key=api_key)

    def _load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the university data.

        Returns:
            pd.DataFrame: The processed university programs data.
        """
        try:
            df = pd.read_csv('programs_data_with_links.csv')

            # Clean and convert 'Ücret (KDV Hariç)' to a numeric format
            if 'Ücret (KDV Hariç)' in df.columns:
                df['Ücret (KDV Hariç)'] = (
                    df['Ücret (KDV Hariç)']
                    .replace(r'[^\d.]', '', regex=True)  # Remove any non-digit characters except dot
                    .astype(float)
                )
            else:
                st.warning("Veri çerçevesinde 'Ücret (KDV Hariç)' sütunu bulunamadı.")

            numeric_columns = [
                'Başarı Sırası 2023', 'Taban Puan 2023', 'Kontenjan 2023',
                'Yerleşen 2023'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    st.warning(f"Veri çerçevesinde '{col}' sütunu bulunamadı.")

            return df
        except FileNotFoundError:
            st.error("Veri dosyası bulunamadı: 'programs_data_with_links.csv'. Lütfen dosyanın doğru konumda olduğundan emin olun.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
            return pd.DataFrame()

    def _create_visualizations(self, df: pd.DataFrame):
        """
        Create interactive visualizations for the filtered results.

        Args:
            df (pd.DataFrame): The filtered university programs data.
        """
        if df.empty:
            return

        col1, col2 = st.columns(2)

        with col1:
            if 'Kontenjan 2023' in df.columns:
                kontenjan_values = df['Kontenjan 2023'].fillna(20)
            else:
                st.warning("Veri çerçevesinde 'Kontenjan 2023' sütunu bulunamadı.")
                kontenjan_values = pd.Series([20]*len(df))

            fig_scatter = px.scatter(
                df,
                x='Başarı Sırası 2023',
                y='Üniversite',
                color='Puan Türü',
                size=kontenjan_values,
                hover_data={
                    'Program Adı': True,
                    'Taban Puan 2023': ':.2f',
                    'Kontenjan 2023': ':.0f'
                },
                title='Üniversite ve Başarı Sırası Dağılımı'
            )
            fig_scatter.update_layout(
                height=600,
                xaxis_title="Başarı Sırası",
                yaxis_title="Üniversite",
                showlegend=True,
                legend_title="Puan Türü"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            if 'Üni.Türü' in df.columns:
                uni_type_counts = df['Üni.Türü'].fillna('Belirtilmemiş').value_counts()
            else:
                st.warning("Veri çerçevesinde 'Üni.Türü' sütunu bulunamadı.")
                uni_type_counts = pd.Series()

            if not uni_type_counts.empty:
                fig_pie = px.pie(
                    values=uni_type_counts.values,
                    names=uni_type_counts.index,
                    title='Üniversite Türü Dağılımı',
                    hole=0.4
                )
                fig_pie.update_layout(
                    height=400,
                    showlegend=True,
                    legend_title="Üniversite Türü"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Üniversite türü dağılımı oluşturulamadı.")

        # Box Plot for Taban Puan
        try:
            if {'Puan Türü', 'Taban Puan 2023', 'Üni.Türü'}.issubset(df.columns):
                box_plot_data = df.dropna(subset=['Puan Türü', 'Taban Puan 2023', 'Üni.Türü'])

                if not box_plot_data.empty:
                    fig_box = px.box(
                        box_plot_data,
                        x='Puan Türü',
                        y='Taban Puan 2023',
                        color='Üni.Türü',
                        title='Puan Türlerine Göre Taban Puan Dağılımı'
                    )
                    fig_box.update_layout(
                        height=500,
                        xaxis_title="Puan Türü",
                        yaxis_title="Taban Puan",
                        showlegend=True,
                        legend_title="Üniversite Türü"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.warning("Taban puan dağılımı için yeterli veri bulunmamaktadır.")
            else:
                st.warning("Gerekli sütunlar ('Puan Türü', 'Taban Puan 2023', 'Üni.Türü') bulunamadı.")
        except Exception as e:
            st.warning(f"Taban puan dağılımı oluşturulurken bir hata oluştu: {str(e)}")

        # Summary Statistics
        st.subheader("📊 Özet İstatistikler")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Toplam Program Sayısı", len(df))

        with col2:
            if 'Başarı Sırası 2023' in df.columns:
                avg_ranking = df['Başarı Sırası 2023'].mean()
                st.metric(
                    "Ortalama Başarı Sırası",
                    f"{avg_ranking:,.0f}" if not pd.isna(avg_ranking) else "Veri yok"
                )
            else:
                st.metric("Ortalama Başarı Sırası", "Veri yok")

        with col3:
            if 'Taban Puan 2023' in df.columns:
                avg_points = df['Taban Puan 2023'].mean()
                st.metric(
                    "Ortalama Taban Puan",
                    f"{avg_points:.2f}" if not pd.isna(avg_points) else "Veri yok"
                )
            else:
                st.metric("Ortalama Taban Puan", "Veri yok")

    def _display_results_table(self, df: pd.DataFrame):
        """
        Display the filtered results in an interactive table.

        Args:
            df (pd.DataFrame): The filtered university programs data.
        """
        if df.empty:
            st.warning("Arama kriterlerinize uygun sonuç bulunamadı.")
            return

        display_columns = [
            'Üniversite', 'Program Adı', 'Şehir', 'Puan Türü',
            'Başarı Sırası 2023','Başarı Sırası 2022', 'Taban Puan 2023','Taban Puan 2022','Kontenjan 2023',
            'Üni.Türü','Yıl','Ücret (KDV Hariç)','YKS Net Ort. Link','University Link','YÖP Link'
        ]

        available_display_columns = [col for col in display_columns if col in df.columns]

        if not available_display_columns:
            st.warning("Gösterilecek uygun sütun bulunamadı.")
            return

        st.dataframe(
            df[available_display_columns].style.format({
                'Başarı Sırası 2023': '{:,.0f}',
                'Taban Puan 2023': '{:.2f}',
                'Ücret (KDV Hariç)': '{:,.2f} ₺'
            }),
            use_container_width=True,
            height=400
        )

    def _add_to_search_history(self, question: str, results_count: int):
        """
        Add search query to history with timestamp.

        Args:
            question (str): The user's question.
            results_count (int): Number of results found.
        """
        st.session_state.search_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'question': question,
            'results_count': results_count
        })

    def _show_search_history(self):
        """
        Display search history in the sidebar.
        """
        if st.session_state.search_history:
            st.sidebar.header("Arama Geçmişi")
            for idx, search in enumerate(reversed(st.session_state.search_history[-5:])):
                with st.sidebar.expander(
                    f"{search['timestamp']} ({search['results_count']} sonuç)",
                    expanded=False
                ):
                    st.write(search['question'])

    def _create_filters_sidebar(self) -> FilterCriteria:
        """
        Create sidebar filters for manual filtering.

        Returns:
            FilterCriteria: An instance containing the selected filter criteria.
        """
        st.sidebar.header("Filtreleme Seçenekleri")

        criteria = FilterCriteria()

        # City Filter
        if 'Şehir' in self.df.columns:
            criteria.cities = st.sidebar.multiselect(
                "Şehir Seçin",
                options=sorted(self.df['Şehir'].dropna().unique()),
                key="recommender_city_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Şehir' sütunu bulunamadı.")

        # University Type Filter
        if 'Üni.Türü' in self.df.columns:
            criteria.university_types = st.sidebar.multiselect(
                "Üniversite Türü",
                options=sorted(self.df['Üni.Türü'].dropna().unique()),
                key="recommender_university_type_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Üni.Türü' sütunu bulunamadı.")

        # Score Type Filter
        if 'Puan Türü' in self.df.columns:
            criteria.score_types = st.sidebar.multiselect(
                "Puan Türü",
                options=sorted(self.df['Puan Türü'].dropna().unique()),
                key="recommender_score_type_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Puan Türü' sütunu bulunamadı.")

        # Language Preference
        if 'Program Adı' in self.df.columns:
            # Assuming language information is embedded in 'Program Adı', else adjust accordingly
            # You might need a separate column for language preferences
            language_options = sorted(set(re.findall(r'\b\w+\b', ' '.join(self.df['Program Adı'].dropna().unique()))))
            criteria.language_types = st.sidebar.multiselect(
                "Dil Tercihi",
                options=language_options,
                key="recommender_language_type_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Program Adı' sütunu bulunamadı.")

        # Faculty Type Filter
        if 'Fakülte' in self.df.columns:
            criteria.faculty_types = st.sidebar.multiselect(
                "Fakülte Türü",
                options=sorted(self.df['Fakülte'].dropna().unique()),
                key="recommender_faculty_type_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Fakülte' sütunu bulunamadı.")

        # Scholarship Percentage
        scholarship_input = st.sidebar.text_input(
            "Burs Yüzdesi (%)",
            placeholder="Örneğin: 50",
            key="recommender_scholarship_input"
        )
        if scholarship_input:
            try:
                criteria.scholarship_percentage = int(scholarship_input)
            except ValueError:
                st.sidebar.warning("Geçerli bir burs yüzdesi girin.")

        # Maximum Fee
        fee_input = st.sidebar.text_input(
            "Maksimum Ücret (₺)",
            placeholder="Örneğin: 50000",
            key="recommender_max_fee_input"
        )
        if fee_input:
            try:
                criteria.max_fee = float(fee_input)
            except ValueError:
                st.sidebar.warning("Geçerli bir ücret girin.")

        # Minimum Quota
        quota_input = st.sidebar.text_input(
            "Minimum Kontenjan",
            placeholder="Örneğin: 10",
            key="recommender_min_quota_input"
        )
        if quota_input:
            try:
                criteria.min_quota = int(quota_input)
            except ValueError:
                st.sidebar.warning("Geçerli bir kontenjan değeri girin.")

        # Taban Puan Filters
        min_score_input = st.sidebar.text_input(
            "Minimum Taban Puan",
            placeholder="Örneğin: 400",
            key="recommender_min_score_input"
        )
        if min_score_input:
            try:
                criteria.min_score = float(min_score_input)
            except ValueError:
                st.sidebar.warning("Geçerli bir taban puan girin.")

        max_score_input = st.sidebar.text_input(
            "Maksimum Taban Puan",
            placeholder="Örneğin: 500",
            key="recommender_max_score_input"
        )
        if max_score_input:
            try:
                criteria.max_score = float(max_score_input)
            except ValueError:
                st.sidebar.warning("Geçerli bir taban puan girin.")

        return criteria

    def _apply_filters(self, df: pd.DataFrame, criteria: FilterCriteria) -> pd.DataFrame:
        """
        Apply manual filtering criteria to the dataframe.

        Args:
            df (pd.DataFrame): The current filtered university programs data.
            criteria (FilterCriteria): The manual filter criteria from the sidebar.

        Returns:
            pd.DataFrame: The further filtered university programs data.
        """
        filtered_df = df.copy()
        
        try:
            # Ranking Filters
            if criteria.min_ranking is not None:
                filtered_df = filtered_df[filtered_df['Başarı Sırası 2023'] >= criteria.min_ranking]
            if criteria.max_ranking is not None:
                filtered_df = filtered_df[filtered_df['Başarı Sırası 2023'] <= criteria.max_ranking]

            # Universities Filter
            if criteria.universities:
                pattern = '|'.join(map(re.escape, criteria.universities))
                filtered_df = filtered_df[filtered_df['Üniversite'].str.contains(pattern, case=False, na=False)]

            # Programs Filter
            if criteria.programs:
                pattern = '|'.join(map(re.escape, criteria.programs))
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(pattern, case=False, na=False)]
                logger.info(f"After filtering programs: {len(filtered_df)}")

            # Cities Filter
            if criteria.cities:
                pattern = '|'.join(map(re.escape, criteria.cities))
                filtered_df = filtered_df[filtered_df['Şehir'].str.contains(pattern, case=False, na=False)]

            # Score Types Filter
            if criteria.score_types:
                pattern = '|'.join(map(re.escape, criteria.score_types))
                filtered_df = filtered_df[filtered_df['Puan Türü'].str.contains(pattern, case=False, na=False)]

            # Language Types Filter
            if criteria.language_types:
                language_pattern = '|'.join(map(re.escape, criteria.language_types))
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(language_pattern, case=False, na=False)]

            # Scholarship Percentage Filter
            if criteria.scholarship_percentage is not None:
                scholarship_str = f"%{criteria.scholarship_percentage}"
                filtered_df = filtered_df[filtered_df['Program Adı'].str.contains(scholarship_str, case=False, na=False)]

            # Faculty Types Filter
            if criteria.faculty_types:
                pattern = '|'.join(map(re.escape, criteria.faculty_types))
                filtered_df = filtered_df[filtered_df['Fakülte'].str.contains(pattern, case=False, na=False)]

            # University Types Filter
            if criteria.university_types:
                pattern = '|'.join(map(re.escape, criteria.university_types))
                filtered_df = filtered_df[filtered_df['Üni.Türü'].str.contains(pattern, case=False, na=False)]

            # Maximum Fee Filter
            if criteria.max_fee is not None:
                filtered_df['Ücret (KDV Hariç)'] = pd.to_numeric(filtered_df['Ücret (KDV Hariç)'], errors='coerce').fillna(0)
                filtered_df = filtered_df[filtered_df['Ücret (KDV Hariç)'] <= criteria.max_fee]

            # Minimum Quota Filter
            if criteria.min_quota is not None:
                filtered_df['Kontenjan 2023'] = pd.to_numeric(filtered_df['Kontenjan 2023'], errors='coerce').fillna(0)
                filtered_df = filtered_df[filtered_df['Kontenjan 2023'] >= criteria.min_quota]

            # Taban Puan Filters
            if criteria.min_score is not None:
                filtered_df = filtered_df[filtered_df['Taban Puan 2023'] >= criteria.min_score]
            if criteria.max_score is not None:
                filtered_df = filtered_df[filtered_df['Taban Puan 2023'] <= criteria.max_score]

            # Sorting
            filtered_df = filtered_df.sort_values('Başarı Sırası 2023', ascending=True)
            return filtered_df

        except Exception as e:
            logger.error(f"Error in applying manual filters: {e}")
            return df

    def run(self):
        """
        Run the University Recommender interface.
        """
        st.title("🎓 YKS Tercih Asistanı")
        st.write("""
        Bu asistan, YKS tercihlerinizde size yardımcı olmak için tasarlanmıştır. 
        Sorunuzu doğal bir dille yazın, size en uygun programları bulalım.
        """)

        # Check if data is loaded
        if self.df.empty:
            st.warning("Veri seti boş veya yüklenemedi. Lütfen dosyayı kontrol edin.")
            return

        # Create sidebar filters
        criteria = self._create_filters_sidebar()

        # Main search interface
        question = st.text_input(
            "Sorunuzu yazın:",
            placeholder="Örnek: İstanbul'da başarı sırası 50000'den iyi olan bilgisayar mühendisliği bölümlerini göster"
        )

        if st.button("Ara", type="primary"):
            if not question.strip():
                st.warning("Lütfen bir soru girin.")
                return

            with st.spinner("Sonuçlar hazırlanıyor..."):
                filtered_df, response = self.recommender.process_question(question, self.df)

                # Apply manual filters
                filtered_df = self._apply_filters(filtered_df, criteria)

                # Store results in session state
                st.session_state.filtered_results = filtered_df
                st.session_state.current_response = response

                # Add to search history
                self._add_to_search_history(question, len(filtered_df))

        # Display results if available
        if st.session_state.filtered_results is not None:
            st.header("🔍 Sonuçlar")

            # Display AI response
            if st.session_state.current_response:
                with st.expander("AI Değerlendirmesi", expanded=True):
                    st.write(st.session_state.current_response)

            # Display results table
            self._display_results_table(st.session_state.filtered_results)

            # Create visualizations
            with st.expander("📊 Görselleştirmeler", expanded=True):
                self._create_visualizations(st.session_state.filtered_results)

        # Show search history
        self._show_search_history()

        # Footer
        st.markdown("""
        ---
        💡 **İpucu**: Daha iyi sonuçlar için, tercih kriterlerinizi detaylı bir şekilde belirtin.
        Örneğin: "İstanbul'da devlet üniversitelerinde, başarı sırası 50000'den iyi olan bilgisayar mühendisliği bölümlerini göster"
        """)


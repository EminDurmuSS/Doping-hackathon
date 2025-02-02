# university/finder.py

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass ,field
import re
from .recommender import UniversityRecommenderInterface


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

class UniversityFinder:
    def __init__(self):
        """Initialize the University Finder interface."""
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load university data from a CSV file.

        Returns:
            pd.DataFrame: The loaded university programs data.
        """
        try:
            df = pd.read_csv(r'C:\Users\pc\Desktop\danisiyorum\motiKocApp\programs_data_with_links.csv')
            # Perform any necessary preprocessing here
            return df
        except FileNotFoundError:
            st.error("Veri dosyası bulunamadı: 'programs_data_with_links.csv'. Lütfen dosyanın doğru konumda olduğundan emin olun.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
            return pd.DataFrame()

    def _create_filters_sidebar(self) -> FilterCriteria:
        """
        Create sidebar filters for university search.

        Returns:
            FilterCriteria: An instance containing the selected filter criteria.
        """
        st.sidebar.header("Üniversite Arama Filtreleri")

        criteria = FilterCriteria()

        # City Filter
        if 'Şehir' in self.df.columns:
            criteria.cities = st.sidebar.multiselect(
                "Şehir Seçin",
                options=sorted(self.df['Şehir'].dropna().unique()),
                key="finder_city_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Şehir' sütunu bulunamadı.")

        # University Type Filter
        if 'Üni.Türü' in self.df.columns:
            criteria.university_types = st.sidebar.multiselect(
                "Üniversite Türü",
                options=sorted(self.df['Üni.Türü'].dropna().unique()),
                key="finder_university_type_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Üni.Türü' sütunu bulunamadı.")

        # Program Name Filter
        if 'Program Adı' in self.df.columns:
            criteria.programs = st.sidebar.text_input(
                "Program Adı",
                placeholder="Örneğin: Bilgisayar Mühendisliği",
                key="finder_program_input"
            ).split(',')  # Allow multiple programs separated by commas
            criteria.programs = [prog.strip() for prog in criteria.programs if prog.strip()]
        else:
            st.sidebar.warning("Veri çerçevesinde 'Program Adı' sütunu bulunamadı.")

        # Ranking Range Slider
        if 'Başarı Sırası 2023' in self.df.columns:
            min_ranking = int(self.df['Başarı Sırası 2023'].min()) if not self.df['Başarı Sırası 2023'].isnull().all() else 0
            max_ranking = int(self.df['Başarı Sırası 2023'].max()) if not self.df['Başarı Sırası 2023'].isnull().all() else 100000
            criteria.min_ranking, criteria.max_ranking = st.sidebar.slider(
                "Başarı Sırası Aralığı",
                min_value=min_ranking,
                max_value=max_ranking,
                value=(min_ranking, max_ranking),
                key="finder_ranking_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Başarı Sırası 2023' sütunu bulunamadı.")

        # Score Type Filter
        if 'Puan Türü' in self.df.columns:
            criteria.score_types = st.sidebar.multiselect(
                "Puan Türü",
                options=sorted(self.df['Puan Türü'].dropna().unique()),
                key="finder_score_type_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Puan Türü' sütunu bulunamadı.")

        # Scholarship Percentage
        scholarship_input = st.sidebar.text_input(
            "Burs Yüzdesi (%)",
            placeholder="Örneğin: 50",
            key="finder_scholarship_input"
        )
        if scholarship_input:
            try:
                criteria.scholarship_percentage = int(scholarship_input)
            except ValueError:
                st.sidebar.warning("Geçerli bir burs yüzdesi girin.")

        # Faculty Type Filter
        if 'Fakülte' in self.df.columns:
            criteria.faculty_types = st.sidebar.multiselect(
                "Fakülte Türü",
                options=sorted(self.df['Fakülte'].dropna().unique()),
                key="finder_faculty_type_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Fakülte' sütunu bulunamadı.")

        # Maximum Fee
        fee_input = st.sidebar.text_input(
            "Maksimum Ücret (₺)",
            placeholder="Örneğin: 50000",
            key="finder_max_fee_input"
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
            key="finder_min_quota_input"
        )
        if quota_input:
            try:
                criteria.min_quota = int(quota_input)
            except ValueError:
                st.sidebar.warning("Geçerli bir kontenjan değeri girin.")

        # Language Preference
        if 'Program Adı' in self.df.columns:
            criteria.language_types = st.sidebar.multiselect(
                "Dil Tercihi",
                options=sorted(self.df['Program Adı'].dropna().unique()),  # Assuming language info is in Program Adı
                key="finder_language_type_filter"
            )
        else:
            st.sidebar.warning("Veri çerçevesinde 'Program Adı' sütunu bulunamadı.")

        # Minimum and Maximum Score
        min_score_input = st.sidebar.text_input(
            "Minimum Taban Puan",
            placeholder="Örneğin: 400",
            key="finder_min_score_input"
        )
        if min_score_input:
            try:
                criteria.min_score = float(min_score_input)
            except ValueError:
                st.sidebar.warning("Geçerli bir taban puan girin.")

        max_score_input = st.sidebar.text_input(
            "Maksimum Taban Puan",
            placeholder="Örneğin: 500",
            key="finder_max_score_input"
        )
        if max_score_input:
            try:
                criteria.max_score = float(max_score_input)
            except ValueError:
                st.sidebar.warning("Geçerli bir taban puan girin.")

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
                print("after filtering programs:", len(filtered_df))

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
            print(f"Error in applying filters: {e}")
            return df

    def run(self):
        """
        Run the University Finder interface.
        """
        st.title("🏫 Üniversite Bulucu")
        st.write("""
        Arama kriterlerinize uygun üniversiteleri ve programları bulun.
        """)

        # Check if data is loaded
        if self.df.empty:
            st.warning("Veri seti boş veya yüklenemedi. Lütfen dosyayı kontrol edin.")
            return

        # Create sidebar filters
        criteria = self._create_filters_sidebar()

        # Search button
        if st.button("Ara"):
            with st.spinner("Sonuçlar aranıyor..."):
                filtered_df = self._apply_filters(self.df, criteria)

                if filtered_df.empty:
                    st.warning("Arama kriterlerinize uygun üniversite veya program bulunamadı.")
                else:
                    st.success(f"{len(filtered_df)} sonuç bulundu.")
                    st.dataframe(filtered_df)

        # Footer
        st.markdown("""
        ---
        💡 **İpucu**: Filtrelerinizi daraltarak daha spesifik sonuçlar elde edebilirsiniz.
        """)
def show_university_finder():
    """
    Function to display the University Finder interface.
    """
    api_key = st.secrets.get("GEMINI_API_KEY")  # Ensure your API key is stored securely
    if not api_key:
        st.error("Gemini API key is missing. Please set it in the secrets.")
        return

    interface = UniversityRecommenderInterface(api_key=api_key)
    interface.run()
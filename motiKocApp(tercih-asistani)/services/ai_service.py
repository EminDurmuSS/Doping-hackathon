# services/ai_service.py

import google.generativeai as genai
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import logging
import re

from config.settings import GEMINI_API_KEY, GEMINI_MODEL

# Logger configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        """Initialize the AI service"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
        except Exception as e:
            logger.error(f"Error initializing AI service: {str(e)}")
            raise e

    def analyze_performance(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Performans verilerini analiz eder ve değerlendirme sağlar.
        
        Args:
            data (Dict[str, Any]): Analiz edilecek veriler.
        
        Returns:
            Optional[Dict[str, Any]]: Analiz sonuçları veya None.
        """
        try:
            prompt = f"""
            Kullanıcının çalışma verilerini ve deneme sınavı sonuçlarını analiz ederek genel bir değerlendirme yap.
            Hedef verileri de dikkate alarak aşağıdaki formatta yanıt ver:

            1. Genel Değerlendirme
            2. Güçlü Yönler
            3. Gelişim Alanları
            4. Performans Tahminleri
            5. Öneriler
            6. Aksiyon Planı

            Veriler:
            {json.dumps(data, ensure_ascii=False)}
            """
            
            response = self.model.generate_content(prompt)
            analysis = self._parse_analysis_response(response.text)
            return analysis
        except Exception as e:
            logger.error(f"Performans analizi yapılırken hata: {str(e)}")
            return None

    def get_subject_recommendations(self, subject: str, stats: Dict[str, Any]) -> List[str]:
        """
        Belirli bir ders için AI tabanlı öneriler sağlar.
        
        Args:
            subject (str): Ders adı.
            stats (Dict[str, Any]): Dersle ilgili istatistikler.
        
        Returns:
            List[str]: Öneriler listesi.
        """
        try:
            prompt = f"""
            Aşağıdaki istatistiklere dayanarak {subject} dersi için öneriler sun:
            {json.dumps(stats, ensure_ascii=False)}
            
            Lütfen şu formatta yanıt ver:
            1. Öneri 1
            2. Öneri 2
            3. Öneri 3
            """
            
            response = self.model.generate_content(prompt)
            recommendations = response.text.strip().split('\n')
            # Önerileri temizleyip liste haline getirin
            recommendations = [rec.strip('- ').strip() for rec in recommendations if rec.strip()]
            return recommendations
        except Exception as e:
            logger.error(f"{subject} dersi için öneri oluşturulurken hata: {str(e)}")
            return []

    def analyze_study_pattern(self, study_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Çalışma verilerini analiz eder ve öneriler sağlar.
        
        Args:
            study_data (List[Dict[str, Any]]): Çalışma verileri listesi.
        
        Returns:
            Dict[str, Any]: Analiz sonuçları.
        """
        try:
            prompt = f"""
            Aşağıdaki çalışma verilerini analiz et ve öneriler sun:
            {json.dumps(study_data, ensure_ascii=False)}
            
            Lütfen şu formatta yanıt ver:
            1. Çalışma Alışkanlıkları Analizi
            2. Gelişim Alanları
            3. Spesifik Öneriler
            4. Gelecek Adımlar
            """
            
            response = self.model.generate_content(prompt)
            analysis = self._parse_analysis_response(response.text)
            return analysis
        except Exception as e:
            logger.error(f"Çalışma alışkanlıkları analizi yapılırken hata: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def generate_study_plan(self, user_data: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kişiselleştirilmiş bir çalışma planı oluşturur.
        
        Args:
            user_data (Dict[str, Any]): Kullanıcı verileri.
            preferences (Dict[str, Any]): Kullanıcı tercihleri.
        
        Returns:
            Dict[str, Any]: Çalışma planı.
        """
        try:
            prompt = f"""
            Aşağıdaki kullanıcı verileri ve tercihlere dayanarak detaylı bir çalışma planı oluştur:
            
            Kullanıcı Verileri:
            {json.dumps(user_data, ensure_ascii=False)}
            
            Tercihler:
            {json.dumps(preferences, ensure_ascii=False)}
            
            Lütfen şu formatta yanıt ver:
            1. Haftalık Program
            2. Günlük Dağılım
            3. Mola Zamanları
            4. Tekrar Dönemleri
            """
            
            response = self.model.generate_content(prompt)
            plan = self._parse_analysis_response(response.text)
            return {
                'plan': plan,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Çalışma planı oluşturulurken hata: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def explain_solution(self, question: str, subject: str) -> Dict[str, Any]:
        """
        Belirli bir sorunun detaylı çözümünü sağlar.
        
        Args:
            question (str): Soru metni.
            subject (str): Ders adı.
        
        Returns:
            Dict[str, Any]: Çözüm açıklaması.
        """
        try:
            prompt = f"""
            Lütfen aşağıdaki {subject} sorusu için detaylı adım adım çözüm sun:
            
            Soru:
            {question}
            
            İçermelidir:
            1. Problem Analizi
            2. Adım Adım Çözüm
            3. Kullanılan Temel Kavramlar
            4. Kaçınılması Gereken Yaygın Hatalar
            5. Benzer Soru Türleri
            """
            
            response = self.model.generate_content(prompt)
            explanation = response.text
            return {
                'explanation': explanation,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Soru çözümü açıklaması oluşturulurken hata: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def generate_motivational_message(self, user_stats: Dict[str, Any]) -> str:
        """
        Kişiye özel motivasyon mesajı oluşturur.
        
        Args:
            user_stats (Dict[str, Any]): Kullanıcı istatistikleri.
        
        Returns:
            str: Motivasyon mesajı.
        """
        try:
            prompt = f"""
            Aşağıdaki istatistiklere dayanarak kişiye özel motivasyon mesajı oluştur:
            {json.dumps(user_stats, ensure_ascii=False)}
            
            Mesaj kişisel, teşvik edici ve ilerlemeye özgü olmalıdır.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            # Hata durumunda varsayılan mesaj
            return "Çalışmalarına devam et! Her gün yeni bir başarıya bir adım daha yakınsın! 💪"

    def analyze_mock_exam(self, exam_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deneme sınavı sonuçlarını analiz eder ve geri bildirim sağlar.
        
        Args:
            exam_results (Dict[str, Any]): Deneme sınavı sonuçları.
        
        Returns:
            Dict[str, Any]: Analiz sonuçları.
        """
        try:
            prompt = f"""
            Aşağıdaki deneme sınavı sonuçlarını analiz et ve detaylı geri bildirim sağla:
            {json.dumps(exam_results, ensure_ascii=False)}
            
            Lütfen şu formatta yanıt ver:
            1. Performans Analizi
            2. Güçlü Alanlar
            3. Gelişim Alanları
            4. Spesifik Çalışma Önerileri
            5. Hedef Puan Projeksiyonları
            """
            
            response = self.model.generate_content(prompt)
            analysis = self._parse_analysis_response(response.text)
            return analysis
        except Exception as e:
            logger.error(f"Deneme sınavı analizi yapılırken hata: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def analyze_career_responses(self, responses: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze career questionnaire responses and provide career analysis.

        Args:
            responses (Dict[str, Any]): User's responses to the career questionnaire.

        Returns:
            Optional[Dict[str, Any]]: Structured analysis results or None if an error occurs.
        """
        try:
            prompt = f"""
            Kullanıcının kariyer anketi yanıtlarını analiz ederek aşağıdaki formatta detaylı bir kariyer analizi yap:

            1. Personality Traits: Kişilik Profili
            2. Strengths: Güçlü Yönler
            3. Development Areas: Gelişim Alanları
            4. Recommended Careers: Kariyer Önerileri
            5. Education Recommendations: Eğitim Önerileri
            6. Next Steps: Sonraki Adımlar

            Yanıtlar:
            {json.dumps(responses, ensure_ascii=False)}

            Lütfen yalnızca aşağıdaki JSON formatında yanıt ver. Başka herhangi bir metin ekleme:

            {{
                "personality_traits": {{
                    "Analitik": int,
                    "Sosyal": int,
                    "Yaratıcılık": int,
                    "Liderlik": int,
                    "Teknik": int,
                    "Uyumluluk": int,
                    "Girişimcilik": int
                }},
                "strengths": [str, str, str],
                "development_areas": [str, str, str],
                "recommended_careers": [
                    {{
                        "title": str,
                        "match_percentage": int,
                        "description": str,
                        "required_skills": [str, str, str],
                        "education_requirements": [str, str, str],
                        "career_path": [str, str, str]
                    }},
                    ...
                ],
                "education_recommendations": [
                    {{
                        "program": str,
                        "description": str,
                        "universities": [str, str, str],
                        "key_courses": [str, str, str]
                    }},
                    ...
                ],
                "next_steps": [str, str, str]
            }}
            """

            response = self.model.generate_content(prompt)
            analysis = self._parse_analysis_response(response.text)
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing career responses: {str(e)}")
            return None

    def generate_career_recommendations(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Generate career recommendations based on user data.

        Args:
            data (Dict[str, Any]): Data including assessment, profile, and performance.

        Returns:
            Optional[List[Dict[str, Any]]]: List of career recommendations or None if an error occurs.
        """
        try:
            prompt = f"""
            Aşağıdaki kullanıcı verilerini analiz ederek detaylı kariyer önerileri oluştur. Lütfen yalnızca aşağıdaki JSON formatında yanıt ver. Başka herhangi bir metin ekleme:

            [
                {{
                    "title": str,
                    "match_percentage": int,
                    "category": str,
                    "description": str,
                    "required_skills": [str, str, str],
                    "education": [str, str, str],
                    "salary_range": {{
                        "min": int,
                        "max": int
                    }},
                    "growth_potential": int,
                    "job_postings": int,
                    "demand_level": str,
                    "competition_level": str,
                    "career_path": [str, str, str],
                    "success_factors": [str, str, str]
                }},
                ...
            ]

            Kullanıcı Verileri:
            {json.dumps(data, ensure_ascii=False)}
            """
            
            response = self.model.generate_content(prompt)
            recommendations = self._parse_recommendations_response(response.text)
            return recommendations
        except Exception as e:
            logger.error(f"Error generating career recommendations: {str(e)}")
            return None

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the AI-generated response into a structured dictionary.

        Args:
            response_text (str): Raw text response from the AI model.

        Returns:
            Dict[str, Any]: Parsed analysis data.
        """
        try:
            # Attempt to parse the response as JSON
            analysis = json.loads(response_text)
            return analysis
        except json.JSONDecodeError:
            # Attempt to extract JSON from code blocks
            try:
                # Regex to extract JSON from code blocks
                json_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
                match = json_pattern.search(response_text)
                if match:
                    return json.loads(match.group(1))
                
                # Fallback: find the first JSON object in the text
                curly_pattern = re.compile(r'\{.*\}', re.DOTALL)
                match = curly_pattern.search(response_text)
                if match:
                    return json.loads(match.group(0))
                
                # If no JSON found, return empty dict
                logger.error("No valid JSON found in AI response.")
                return {}
            except Exception as e:
                logger.error(f"Error parsing analysis response: {str(e)}")
                return {}
    
    def _parse_recommendations_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the AI-generated career recommendations response into a list of dictionaries.

        Args:
            response_text (str): Raw text response from the AI model.

        Returns:
            List[Dict[str, Any]]: List of career recommendations.
        """
        try:
            # Attempt to parse the response as JSON
            recommendations = json.loads(response_text)
            return recommendations
        except json.JSONDecodeError:
            # Attempt to extract JSON from code blocks
            try:
                # Regex to extract JSON from code blocks
                json_pattern = re.compile(r'```(?:json)?\s*(\[\{.*\}\])\s*```', re.DOTALL)
                match = json_pattern.search(response_text)
                if match:
                    return json.loads(match.group(1))
                
                # Fallback: find the first JSON array in the text
                array_pattern = re.compile(r'\[\{.*\}\]', re.DOTALL)
                match = array_pattern.search(response_text)
                if match:
                    return json.loads(match.group(0))
                
                # If no JSON found, return empty list
                logger.error("No valid JSON array found in AI response.")
                return []
            except Exception as e:
                logger.error(f"Error parsing career recommendations response: {str(e)}")
                return []
    
# features/forum/views.py

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
from .models import ForumPost, ForumComment, ForumCategory
from services.ai_service import AIService
import sqlite3
import base64
from io import BytesIO
import time

class ForumView:
    def __init__(self, user_id: int, db_path: str = 'motikoc.db'):
        self.user_id = user_id
        self.db_conn = sqlite3.connect(db_path)
        self.db_conn.row_factory = sqlite3.Row  # Enable dictionary-like cursor
        self.ai_service = AIService()
        self.init_forum_tables()

    def init_forum_tables(self):
        """Initialize forum tables if they do not exist."""
        try:
            cursor = self.db_conn.cursor()
            # Forum categories
            cursor.execute('''CREATE TABLE IF NOT EXISTS forum_categories
                             (id INTEGER PRIMARY KEY,
                              name TEXT UNIQUE,
                              description TEXT,
                              icon TEXT,
                              order_index INTEGER)''')
            
            # Forum questions
            cursor.execute('''CREATE TABLE IF NOT EXISTS forum_questions
                             (id INTEGER PRIMARY KEY,
                              user_id INTEGER,
                              category_id INTEGER,
                              title TEXT,
                              content TEXT,
                              tags TEXT,
                              created_at TEXT,
                              updated_at TEXT,
                              view_count INTEGER DEFAULT 0,
                              is_solved BOOLEAN DEFAULT FALSE,
                              FOREIGN KEY(user_id) REFERENCES users(id),
                              FOREIGN KEY(category_id) REFERENCES forum_categories(id))''')
            
            # Forum answers
            cursor.execute('''CREATE TABLE IF NOT EXISTS forum_answers
                             (id INTEGER PRIMARY KEY,
                              question_id INTEGER,
                              user_id INTEGER,
                              content TEXT,
                              created_at TEXT,
                              updated_at TEXT,
                              is_accepted BOOLEAN DEFAULT FALSE,
                              upvotes INTEGER DEFAULT 0,
                              FOREIGN KEY(question_id) REFERENCES forum_questions(id),
                              FOREIGN KEY(user_id) REFERENCES users(id))''')
            
            # Forum votes
            cursor.execute('''CREATE TABLE IF NOT EXISTS forum_votes
                             (id INTEGER PRIMARY KEY,
                              user_id INTEGER,
                              content_type TEXT,
                              content_id INTEGER,
                              vote_type INTEGER,
                              created_at TEXT,
                              FOREIGN KEY(user_id) REFERENCES users(id))''')
            
            # Forum notifications
            cursor.execute('''CREATE TABLE IF NOT EXISTS forum_notifications
                             (id INTEGER PRIMARY KEY,
                              user_id INTEGER,
                              content TEXT,
                              link TEXT,
                              created_at TEXT,
                              is_read BOOLEAN DEFAULT FALSE,
                              FOREIGN KEY(user_id) REFERENCES users(id))''')
            
            # Insert default categories if table is empty
            cursor.execute("SELECT COUNT(*) FROM forum_categories")
            if cursor.fetchone()[0] == 0:
                default_categories = [
                    ('TYT Matematik', 'TYT matematik konuları ve soru çözümleri', '📐', 1),
                    ('TYT Türkçe', 'TYT Türkçe ve dil bilgisi konuları', '📚', 2),
                    ('TYT Fen Bilimleri', 'TYT Fizik, Kimya ve Biyoloji', '🔬', 3),
                    ('TYT Sosyal Bilimler', 'TYT Tarih, Coğrafya ve Felsefe', '🌍', 4),
                    ('AYT Matematik', 'AYT matematik konuları ve soru çözümleri', '🧮', 5),
                    ('AYT Fizik', 'AYT fizik konuları ve problemler', '⚡', 6),
                    ('AYT Kimya', 'AYT kimya konuları ve deneyler', '⚗️', 7),
                    ('AYT Biyoloji', 'AYT biyoloji konuları', '🧬', 8),
                    ('Genel YKS', 'YKS sınavı hakkında genel konular', '📋', 9),
                    ('Motivasyon', 'Motivasyon ve çalışma teknikleri', '💪', 10)
                ]
                cursor.executemany('''
                    INSERT INTO forum_categories (name, description, icon, order_index)
                    VALUES (?, ?, ?, ?)
                ''', default_categories)
                self.db_conn.commit()
        except Exception as e:
            st.error(f"Forum tablolarını başlatırken bir hata oluştu: {str(e)}")
            self.db_conn.rollback()

    def show(self):
        """Main method to display the forum interface."""
        st.title("💬 YKS Forumu")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Tüm Sorular",
            "Soru Sor",
            "Kategoriler",
            "Arama"
        ])
        
        with tab1:
            self._show_all_questions()
        
        with tab2:
            self._show_ask_form()
        
        with tab3:
            self._show_categories()
        
        with tab4:
            self._show_search()

        # Display user notifications in the sidebar
        self.show_user_notifications()

    def _show_all_questions(self):
        """Display all forum questions with filtering options."""
        st.markdown("### 📝 Tüm Sorular")
        
        # Filtering options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sort_by = st.selectbox(
                "Sıralama",
                ["En Yeni", "En Çok Cevaplanan", "En Çok Görüntülenen"],
                key="sort_by_all_questions"
            )
        
        with col2:
            filter_solved = st.selectbox(
                "Durum",
                ["Tümü", "Çözülmüş", "Çözülmemiş"],
                key="filter_solved_all_questions"
            )
        
        with col3:
            selected_category = st.selectbox(
                "Kategori",
                ["Tümü"] + self._get_category_names(),
                key="selected_category_all_questions"
            )
        
        # Get and display questions
        questions = self._get_filtered_questions(
            sort_by,
            filter_solved,
            selected_category
        )
        
        if questions:
            for question in questions:
                self._display_question_card(question)
        else:
            st.info("Henüz soru bulunmuyor veya seçilen kriterlere uygun soru yok.")

    def _show_ask_form(self):
        """Display the form to ask a new question."""
        st.markdown("### ❓ Yeni Soru Sor")
        
        with st.form("ask_question_form"):
            category = st.selectbox(
                "Kategori",
                self._get_category_names(),
                key="ask_question_category"
            )
            
            title = st.text_input(
                "Başlık",
                max_chars=100,
                key="ask_question_title"
            )
            
            content = st.text_area(
                "Soru İçeriği",
                height=200,
                key="ask_question_content"
            )
            
            tags = st.text_input(
                "Etiketler (virgülle ayırın)",
                placeholder="matematik, türev, limit",
                key="ask_question_tags"
            )
            
            submitted = st.form_submit_button("Soruyu Gönder")
            
            if submitted:
                if not title.strip() or not content.strip():
                    st.error("Başlık ve soru içeriği zorunludur!")
                else:
                    success = self._save_question({
                        'category': self._get_category_id_by_name(category),
                        'title': title.strip(),
                        'content': content.strip(),
                        'tags': [tag.strip() for tag in tags.split(',')] if tags else []
                    })
                    
                    if success:
                        st.success("Sorunuz başarıyla gönderildi!")
                        st.rerun()
                    else:
                        st.error("Soru gönderilirken bir hata oluştu!")

    def _show_categories(self):
        """Display all forum categories."""
        st.markdown("### 📚 Kategoriler")
        
        categories = self._get_categories()
        
        if categories:
            col1, col2 = st.columns(2)
            
            for i, category in enumerate(categories):
                with col1 if i % 2 == 0 else col2:
                    with st.container():
                        st.markdown(f"""
                        <div style='padding: 15px; 
                                  background-color: rgba(255, 255, 255, 0.05);
                                  border-radius: 10px;
                                  margin-bottom: 10px;'>
                            <h4>{category.icon} {category.name}</h4>
                            <p>{category.description}</p>
                            <small>Soru Sayısı: {self._get_question_count(category.id)}</small>
                            <br/>
                            <button onclick="window.location.href='/?page=forum&action=view_category&category_id={category.id}'"
                                    style="padding: 5px 10px; margin-top: 10px; border: none; border-radius: 5px; background-color: #4CAF50; color: white; cursor: pointer;">
                                Soruları Gör
                            </button>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Kategori bulunamadı.")

    def _show_search(self):
        """Display the search interface for the forum."""
        st.markdown("### 🔍 Forum Ara")
        
        # Search form
        query = st.text_input("Arama", key="forum_search_query")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_in = st.multiselect(
                "Arama Yeri",
                ["Başlıklar", "İçerik", "Yanıtlar"],
                default=["Başlıklar", "İçerik"],
                key="forum_search_in"
            )
        
        with col2:
            date_range = st.selectbox(
                "Zaman Aralığı",
                ["Tüm Zamanlar", "Son 24 Saat", "Son Hafta", "Son Ay"],
                key="forum_search_date_range"
            )
        
        if st.button("🔍 Ara", key="forum_search_button"):
            if not query.strip():
                st.error("Arama yapmak için bir kelime girin.")
            else:
                results = self._search_forum(
                    query.strip(),
                    search_in,
                    date_range
                )
                
                if results:
                    st.markdown(f"### 🎯 Arama Sonuçları ({len(results)} sonuç)")
                    for result in results:
                        self._display_search_result(result)
                else:
                    st.info("Arama kriterlerinize uygun sonuç bulunamadı.")

    def _display_question_card(self, question: sqlite3.Row):
        """Display a single question card."""
        with st.container():
            st.markdown(f"""
            <div style='padding: 15px; 
                      background-color: rgba(255, 255, 255, 0.05);
                      border-radius: 10px;
                      margin-bottom: 10px;'>
                <h4>{question['title']}</h4>
                <p>{question['content'][:200]}...</p>
                <div style='display: flex; justify-content: space-between;'>
                    <small>
                        👤 {question['username']} | 
                        💬 {question['answer_count']} yanıt | 
                        👁️ {question['view_count']} görüntülenme
                    </small>
                    <small>{question['created_at']}</small>
                </div>
                <button onclick="window.location.href='/?page=forum&action=view_question&question_id={question['id']}'"
                        style="padding: 5px 10px; margin-top: 10px; border: none; border-radius: 5px; background-color: #008CBA; color: white; cursor: pointer;">
                    Detayları Gör
                </button>
            </div>
            """, unsafe_allow_html=True)

    def _display_search_result(self, result: sqlite3.Row):
        """Display a single search result."""
        with st.container():
            st.markdown(f"""
            <div style='padding: 15px; 
                      background-color: rgba(255, 255, 255, 0.05);
                      border-radius: 10px;
                      margin-bottom: 10px;'>
                <h4>{result['title']}</h4>
                <p>{result['content'][:200]}...</p>
                <small>
                    {result['category_name']} | 
                    {result['created_at']} | 
                    {result['answer_count']} yanıt
                </small>
                <button onclick="window.location.href='/?page=forum&action=view_question&question_id={result['id']}'"
                        style="padding: 5px 10px; margin-top: 10px; border: none; border-radius: 5px; background-color: #008CBA; color: white; cursor: pointer;">
                    Detayları Gör
                </button>
            </div>
            """, unsafe_allow_html=True)

    def _get_filtered_questions(
        self,
        sort_by: str,
        filter_solved: str,
        category: str
    ) -> List[sqlite3.Row]:
        """Retrieve filtered forum questions based on sorting and filtering criteria."""
        try:
            cursor = self.db_conn.cursor()
            
            # Build base query
            query = '''
                SELECT q.*, u.username, c.name as category_name,
                       COUNT(a.id) as answer_count
                FROM forum_questions q
                JOIN users u ON q.user_id = u.id
                JOIN forum_categories c ON q.category_id = c.id
                LEFT JOIN forum_answers a ON q.id = a.question_id
                WHERE 1=1
            '''
            params = []
            
            # Apply filters
            if filter_solved != "Tümü":
                query += " AND q.is_solved = ?"
                params.append(filter_solved == "Çözülmüş")
            
            if category != "Tümü":
                query += " AND c.name = ?"
                params.append(category)
            
            # Group by question ID
            query += " GROUP BY q.id"
            
            # Apply sorting
            if sort_by == "En Çok Cevaplanan":
                query += " ORDER BY answer_count DESC"
            elif sort_by == "En Çok Görüntülenen":
                query += " ORDER BY q.view_count DESC"
            else:  # En Yeni
                query += " ORDER BY q.created_at DESC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            return results
        except Exception as e:
            st.error(f"Sorular alınırken bir hata oluştu: {str(e)}")
            return []

    def _get_category_names(self) -> List[str]:
        """Retrieve all category names from the database."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT name FROM forum_categories ORDER BY order_index")
            categories = cursor.fetchall()
            return [category['name'] for category in categories]
        except Exception as e:
            st.error(f"Kategori isimleri alınırken bir hata oluştu: {str(e)}")
            return []

    def _get_category_id_by_name(self, name: str) -> Optional[int]:
        """Get the category ID based on its name."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT id FROM forum_categories WHERE name = ?", (name,))
            result = cursor.fetchone()
            if result and 'id' in result:
                return result['id']
            return None  # Explicitly return None if no result is found
        except Exception as e:
            st.error(f"Kategori ID'si alınırken bir hata oluştu: {str(e)}")
            return None


    def _get_question_count(self, category_id: Optional[int]) -> int:
        """Get the count of questions in a specific category."""
        if category_id is None:
            return 0  # Return a default count if category_id is None
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM forum_questions 
                WHERE category_id = ?
            """, (category_id,))
            result = cursor.fetchone()
            return result['count'] if result else 0
        except Exception as e:
            st.error(f"Soru sayısı alınırken bir hata oluştu: {str(e)}")
            return 0


    def _save_question(self, question_data: Dict[str, Any]) -> bool:
        """Save a new question to the database."""
        try:
            cursor = self.db_conn.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO forum_questions 
                (user_id, category_id, title, content, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.user_id,
                question_data['category'],
                question_data['title'],
                question_data['content'],
                ','.join(question_data['tags']),
                now,
                now
            ))
            self.db_conn.commit()
            return True
        except Exception as e:
            st.error(f"Soru kaydedilirken bir hata oluştu: {str(e)}")
            self.db_conn.rollback()
            return False

    def _get_categories(self) -> List[ForumCategory]:
        """Retrieve all forum categories as ForumCategory objects."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT * FROM forum_categories ORDER BY order_index")
            rows = cursor.fetchall()
            return [ForumCategory.from_db_row(row) for row in rows]
        except Exception as e:
            st.error(f"Kategoriler alınırken bir hata oluştu: {str(e)}")
            return []

    def _search_forum(self, query: str, search_in: List[str], date_range: str) -> List[sqlite3.Row]:
        """Search the forum based on query, search locations, and date range."""
        try:
            cursor = self.db_conn.cursor()
            base_query = '''
                SELECT q.*, u.username, c.name as category_name,
                       COUNT(a.id) as answer_count
                FROM forum_questions q
                JOIN users u ON q.user_id = u.id
                JOIN forum_categories c ON q.category_id = c.id
                LEFT JOIN forum_answers a ON q.id = a.question_id
                WHERE 1=1
            '''
            params = []

            # Search conditions
            search_conditions = []
            if 'Başlıklar' in search_in:
                search_conditions.append("q.title LIKE ?")
                params.append(f"%{query}%")
            if 'İçerik' in search_in:
                search_conditions.append("q.content LIKE ?")
                params.append(f"%{query}%")
            if 'Yanıtlar' in search_in:
                search_conditions.append("""
                    EXISTS (
                        SELECT 1 FROM forum_answers a 
                        WHERE a.question_id = q.id AND a.content LIKE ?
                    )
                """)
                params.append(f"%{query}%")
            
            if search_conditions:
                base_query += " AND (" + " OR ".join(search_conditions) + ")"

            # Date range conditions
            if date_range == "Son 24 Saat":
                base_query += " AND q.created_at >= datetime('now', '-1 day')"
            elif date_range == "Son Hafta":
                base_query += " AND q.created_at >= datetime('now', '-7 days')"
            elif date_range == "Son Ay":
                base_query += " AND q.created_at >= datetime('now', '-1 month')"
            # "Tüm Zamanlar" requires no additional filter

            # Group by question ID
            base_query += " GROUP BY q.id"

            # Execute the query
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            return results
        except Exception as e:
            st.error(f"Forum araması yapılırken bir hata oluştu: {str(e)}")
            return []

    def show_question_detail(self, question_id: int):
        """Display the detailed view of a specific question, including answers."""
        try:
            cursor = self.db_conn.cursor()
            
            # Increment view count
            cursor.execute("""
                UPDATE forum_questions 
                SET view_count = view_count + 1 
                WHERE id = ?
            """, (question_id,))
            self.db_conn.commit()
            
            # Fetch question details
            cursor.execute("""
                SELECT q.*, u.username, c.name as category_name, c.icon as category_icon
                FROM forum_questions q
                JOIN users u ON q.user_id = u.id
                JOIN forum_categories c ON q.category_id = c.id
                WHERE q.id = ?
            """, (question_id,))
            question = cursor.fetchone()
            
            if not question:
                st.error("Soru bulunamadı!")
                return
            
            # Display question details
            st.markdown(f"""
            <div style='
                padding: 20px;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.05);
                margin-bottom: 20px;
            '>
                <h2>{question['title']}</h2>
                <div style='
                    display: flex;
                    justify-content: space-between;
                    color: #B0B0B0;
                    margin-top: 10px;
                '>
                    <span>👤 {question['username']} | 💬 {question['is_solved'] and "Çözülmüş" or "Çözülmemiş"} | 👁️ {question['view_count']} görüntülenme</span>
                    <span>📅 {question['created_at']}</span>
                </div>
                <p style='margin-top: 20px;'>{question['content']}</p>
                <p><strong>Etiketler:</strong> {self._format_tags(question['tags'])}</p>
                <div style='margin-top: 10px;'>
                    <button onclick="window.location.href='/?page=forum&action=vote&content_type=question&content_id={question['id']}&vote_type=1'"
                            style="padding: 5px 10px; margin-right: 5px; border: none; border-radius: 5px; background-color: #4CAF50; color: white; cursor: pointer;">
                        👍 Beğen
                    </button>
                    <button onclick="window.location.href='/?page=forum&action=vote&content_type=question&content_id={question['id']}&vote_type=-1'"
                            style="padding: 5px 10px; border: none; border-radius: 5px; background-color: #f44336; color: white; cursor: pointer;">
                        👎 Beğenme
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display answers
            self.show_answers(question_id)
            
            # Answer submission form
            self.show_answer_form(question_id)
            
            # Option to accept answer if user is the question owner
            if question['user_id'] == self.user_id:
                self.show_accept_answer_option(question_id)
            
        except Exception as e:
            st.error(f"Soru detayları gösterilirken bir hata oluştu: {str(e)}")

    def _format_tags(self, tags: Optional[str]) -> str:
        """Format tags for display."""
        if not tags:
            return "Yok"
        tag_list = [f"`{tag.strip()}`" for tag in tags.split(',')]
        return ' '.join(tag_list)

    def show_answers(self, question_id: int):
        """Display all answers for a given question."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT a.*, u.username,
                       (SELECT COUNT(*) FROM forum_votes WHERE content_type = 'answer' AND content_id = a.id AND vote_type = 1) as upvotes,
                       (SELECT COUNT(*) FROM forum_votes WHERE content_type = 'answer' AND content_id = a.id AND vote_type = -1) as downvotes
                FROM forum_answers a
                JOIN users u ON a.user_id = u.id
                WHERE a.question_id = ?
                ORDER BY a.is_accepted DESC, upvotes DESC, a.created_at ASC
            """, (question_id,))
            answers = cursor.fetchall()
            
            st.markdown("### 💬 Cevaplar")
            
            if answers:
                for answer in answers:
                    self._display_answer_card(answer)
            else:
                st.info("Henüz cevap yok. İlk cevabı siz verin!")
        except Exception as e:
            st.error(f"Cevaplar alınırken bir hata oluştu: {str(e)}")

    def _display_answer_card(self, answer: sqlite3.Row):
        """Display a single answer card."""
        accepted_style = "background-color: rgba(0, 255, 0, 0.1);" if answer['is_accepted'] else "background-color: rgba(255, 255, 255, 0.05);"
        
        with st.container():
            st.markdown(f"""
            <div style='padding: 15px; border-radius: 10px; margin: 10px 0; {accepted_style}'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span>👤 {answer['username']} | 📅 {answer['created_at']}</span>
                    <span>👍 {answer['upvotes']} | 👎 {answer['downvotes']}</span>
                </div>
                <p style='margin: 10px 0;'>{answer['content']}</p>
                <div>
                    <button onclick="window.location.href='/?page=forum&action=vote&content_type=answer&content_id={answer['id']}&vote_type=1'"
                            style="padding: 5px 10px; margin-right: 5px; border: none; border-radius: 5px; background-color: #4CAF50; color: white; cursor: pointer;">
                        👍 Beğen
                    </button>
                    <button onclick="window.location.href='/?page=forum&action=vote&content_type=answer&content_id={answer['id']}&vote_type=-1'"
                            style="padding: 5px 10px; border: none; border-radius: 5px; background-color: #f44336; color: white; cursor: pointer;">
                        👎 Beğenme
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def show_answer_form(self, question_id: int):
        """Display the form to submit a new answer."""
        st.markdown("### 💭 Cevap Yaz")
        with st.form(f"answer_form_{question_id}"):
            answer_content = st.text_area(
                "Cevabınız",
                height=150,
                placeholder="Bu soruya cevabınızı yazın...",
                key=f"answer_content_{question_id}"
            )
            
            submitted = st.form_submit_button("Cevabı Gönder")
            
            if submitted:
                if not answer_content.strip():
                    st.error("Cevap içeriği zorunludur!")
                else:
                    success = self._save_answer(question_id, answer_content.strip())
                    if success:
                        st.success("Cevabınız başarıyla gönderildi!")
                        st.rerun()
                    else:
                        st.error("Cevap gönderilirken bir hata oluştu!")

    
    def show_accept_answer_option(self, question_id: int):
        """Provide an option to accept an answer if the user is the question owner."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT a.id, a.user_id, a.content 
                FROM forum_answers a
                WHERE a.question_id = ? AND a.is_accepted = FALSE
            """, (question_id,))
            pending_answers = cursor.fetchall()
            
            if pending_answers:
                st.markdown("---")
                st.markdown("### ✅ Cevap Kabul Et")
                for answer in pending_answers:
                    answer_id = answer['id']
                    if st.button(f"Cevabı Kabul Et ({answer_id})"):
                        self._accept_answer(question_id, answer_id)
                        st.success("Cevap kabul edildi!")
                        st.rerun()
        except Exception as e:
            st.error(f"Cevap kabul edilirken bir hata oluştu: {str(e)}")

    def _accept_answer(self, question_id: int, answer_id: int):
        """Accept an answer as the correct one."""
        try:
            cursor = self.db_conn.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Reset previously accepted answers
            cursor.execute("""
                UPDATE forum_answers 
                SET is_accepted = FALSE 
                WHERE question_id = ?
            """, (question_id,))
            
            # Accept the new answer
            cursor.execute("""
                UPDATE forum_answers 
                SET is_accepted = TRUE 
                WHERE id = ?
            """, (answer_id,))
            
            # Mark the question as solved
            cursor.execute("""
                UPDATE forum_questions 
                SET is_solved = TRUE 
                WHERE id = ?
            """, (question_id,))
            
            self.db_conn.commit()
        except Exception as e:
            st.error(f"Cevap kabul edilirken bir hata oluştu: {str(e)}")
            self.db_conn.rollback()

    def vote_content(self, content_type: str, content_id: int, vote_type: int):
        """
        Handle voting for both questions and answers.
        content_type: 'question' or 'answer'
        vote_type: 1 for upvote, -1 for downvote
        """
        try:
            cursor = self.db_conn.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Validate content_type
            if content_type not in ['question', 'answer']:
                st.error("Geçersiz içerik türü!")
                return
            
            # Check existing vote
            cursor.execute("""
                SELECT id, vote_type 
                FROM forum_votes 
                WHERE user_id = ? AND content_type = ? AND content_id = ?
            """, (self.user_id, content_type, content_id))
            existing_vote = cursor.fetchone()
            
            if existing_vote:
                if existing_vote['vote_type'] == vote_type:
                    # Remove the existing vote
                    cursor.execute("""
                        DELETE FROM forum_votes 
                        WHERE id = ?
                    """, (existing_vote['id'],))
                else:
                    # Update the vote
                    cursor.execute("""
                        UPDATE forum_votes 
                        SET vote_type = ?, created_at = ?
                        WHERE id = ?
                    """, (vote_type, now, existing_vote['id']))
            else:
                # Insert a new vote
                cursor.execute("""
                    INSERT INTO forum_votes 
                    (user_id, content_type, content_id, vote_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (self.user_id, content_type, content_id, vote_type, now))
            
            self.db_conn.commit()
            st.success("Oylama başarıyla güncellendi!")
            st.rerun()
        except Exception as e:
            st.error(f"Oylama işlemi sırasında bir hata oluştu: {str(e)}")
            self.db_conn.rollback()

    def show_user_notifications(self):
        """Display user notifications in the sidebar."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT * FROM forum_notifications
                WHERE user_id = ? AND is_read = FALSE
                ORDER BY created_at DESC
            """, (self.user_id,))
            notifications = cursor.fetchall()
            
            if notifications:
                st.sidebar.markdown("### 🔔 Bildirimler")
                for notif in notifications:
                    with st.sidebar.container():
                        st.markdown(f"""
                        <div style='padding: 10px; border-radius: 5px; 
                                  background-color: rgba(255, 255, 255, 0.05);'>
                            {notif['content']}
                            <br/>
                            <small>{notif['created_at']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.sidebar.columns([1, 1])
                        
                        with col1:
                            if st.sidebar.button("Okundu", key=f"read_notification_{notif['id']}"):
                                self._mark_notification_read(notif['id'])
                                st.rerun()
                        
                        with col2:
                            if notif['link']:
                                if st.sidebar.button("Git", key=f"goto_notification_{notif['id']}"):
                                    # Redirect to the related question
                                    st.session_state.current_question = int(notif['link'])
                                    st.session_state.current_page = "question_detail"
                                    st.rerun()
        except Exception as e:
            st.error(f"Bildirimler yüklenirken bir hata oluştu: {str(e)}")

    def _mark_notification_read(self, notification_id: int):
        """Mark a notification as read."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                UPDATE forum_notifications
                SET is_read = TRUE
                WHERE id = ? AND user_id = ?
            """, (notification_id, self.user_id))
            self.db_conn.commit()
        except Exception as e:
            st.error(f"Bildirim güncellenirken bir hata oluştu: {str(e)}")
            self.db_conn.rollback()

    def _save_answer(self, question_id: int, content: str) -> bool:
        """Save a new answer to the database."""
        try:
            cursor = self.db_conn.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO forum_answers 
                (question_id, user_id, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                question_id,
                self.user_id,
                content,
                now,
                now
            ))
            self.db_conn.commit()
            return True
        except Exception as e:
            st.error(f"Cevap kaydedilirken bir hata oluştu: {str(e)}")
            self.db_conn.rollback()
            return False

    # Additional helper methods can be added here as needed
def show_forum():
    """
    Function to initialize and display the forum.
    This function is intended to be imported and called from main.py or other modules.
    """
    if 'user_id' not in st.session_state or st.session_state.user_id is None:
        st.error("Kullanıcı oturumu açık değil. Lütfen giriş yapın.")
        return

    user_id = st.session_state.user_id
    forum_view = ForumView(user_id)
    forum_view.show()
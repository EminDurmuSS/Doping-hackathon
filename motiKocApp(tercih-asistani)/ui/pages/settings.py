import streamlit as st
from typing import Optional, Dict, Any, Union, cast
from datetime import datetime
import hashlib
from sqlite3 import Row

from ..components.cards import AlertCard
from core.database import DatabaseManager, DatabaseError, db_transaction
from utils.validators import (
    validate_email,
    validate_password,
    validate_profile_data
)

class SettingsPage:
    def __init__(self, user_id: Optional[int] = None):
        self.user_id = user_id
        self.db_manager = DatabaseManager()

    def show(self):
        """Display settings page"""
        st.title("⚙️ Ayarlar")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Profil",
            "Çalışma Tercihleri",
            "Bildirimler",
            "Uygulama"
        ])
        
        with tab1:
            self._show_profile_settings()
        
        with tab2:
            self._show_study_preferences()
        
        with tab3:
            self._show_notification_settings()
        
        with tab4:
            self._show_app_settings()

    def _row_to_dict(self, row: Optional[Row]) -> Dict[str, Any]:
        """Convert sqlite3.Row to a dictionary."""
        if row is None:
            return {}
        return {key: row[key] for key in row.keys()} if isinstance(row, Row) else {}

    def _show_profile_settings(self):
        """Display profile settings section"""
        st.markdown("### 👤 Profil Ayarları")
        
        profile_data = self._get_profile_data()
        
        with st.form("profile_settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(
                    "Ad Soyad",
                    value=profile_data.get('name', ''),
                    key="name_setting"
                )
                
                email = st.text_input(
                    "E-posta",
                    value=profile_data.get('email', ''),
                    key="email_setting"
                )
                
                city = st.text_input(
                    "Şehir",
                    value=profile_data.get('city', ''),
                    key="city_setting"
                )
            
            with col2:
                grade_options = ["9. Sınıf", "10. Sınıf", "11. Sınıf", "12. Sınıf", "Mezun"]
                grade = st.selectbox(
                    "Sınıf Düzeyi*",
                    grade_options,
                    index=grade_options.index(profile_data.get('grade', "9. Sınıf")),
                    key="grade_setting"
                )

                study_type_options = ["Sayısal", "Eşit Ağırlık", "Sözel", "Dil"]
                study_type = st.selectbox(
                    "Alan*",
                    study_type_options,
                    index=study_type_options.index(profile_data.get('study_type', "Sayısal")),
                    key="study_type_setting"
                )
            
            st.markdown("### 🎯 Hedef Bilgileri")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_university = st.text_input(
                    "Hedef Üniversite",
                    value=profile_data.get('target_university', ''),
                    key="target_uni_setting"
                )
                
                target_department = st.text_input(
                    "Hedef Bölüm",
                    value=profile_data.get('target_department', ''),
                    key="target_dept_setting"
                )
            
            with col2:
                target_rank = st.number_input(
                    "Hedef Sıralama",
                    min_value=1,
                    value=profile_data.get('target_rank', 10000),
                    key="target_rank_setting"
                )
            
            # Password change section
            st.markdown("### 🔐 Şifre Değişikliği")
            
            current_password = st.text_input(
                "Mevcut Şifre",
                type="password",
                key="current_password"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_password = st.text_input(
                    "Yeni Şifre",
                    type="password",
                    key="new_password"
                )
            
            with col2:
                confirm_password = st.text_input(
                    "Yeni Şifre (Tekrar)",
                    type="password",
                    key="confirm_password"
                )
            
            submit = st.form_submit_button("Kaydet")
            
            if submit:
                # Validate form data
                profile_update = {
                    'name': name,
                    'email': email,
                    'city': city,
                    'grade': grade,
                    'study_type': study_type,
                    'target_university': target_university,
                    'target_department': target_department,
                    'target_rank': target_rank
                }
                
                validation = validate_profile_data(profile_update)
                
                if not validation['valid']:
                    for message in validation['messages']:
                        AlertCard.error(message)
                    return
                
                # Handle password change if provided
                if new_password:
                    if new_password != confirm_password:
                        AlertCard.error("Yeni şifreler eşleşmiyor!")
                        return
                    
                    # Validate new password
                    password_validation = validate_password(new_password)
                    if not password_validation['valid']:
                        for message in password_validation['messages']:
                            AlertCard.error(message)
                        return
                    
                    # Verify current password
                    if not self._verify_password(current_password):
                        AlertCard.error("Mevcut şifre yanlış!")
                        return
                    
                    # Update password
                    self._update_password(new_password)
                
                # Update profile
                if self._update_profile(profile_update):
                    AlertCard.success("Profil başarıyla güncellendi!")
                    st.rerun()
                else:
                    AlertCard.error("Profil güncellenirken bir hata oluştu!")

    def _show_study_preferences(self):
        """Display study preferences section"""
        st.markdown("### 📚 Çalışma Tercihleri")
        
        preferences = self._get_study_preferences()
        
        with st.form("study_preferences"):
            # Daily study goals
            st.markdown("#### 🎯 Günlük Hedefler")
            
            col1, col2 = st.columns(2)
            
            with col1:
                daily_hours = st.number_input(
                    "Günlük Çalışma Hedefi (Saat)",
                    min_value=1,
                    max_value=12,
                    value=preferences.get('daily_hours', 3),
                    key="daily_hours_setting"
                )
            
            with col2:
                daily_questions = st.number_input(
                    "Günlük Soru Hedefi",
                    min_value=10,
                    max_value=300,
                    value=preferences.get('daily_questions', 50),
                    key="daily_questions_setting"
                )
            
            # Study schedule
            st.markdown("#### ⏰ Çalışma Programı")
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_time = st.time_input(
                    "Başlangıç Saati",
                    value=datetime.strptime(
                        preferences.get('start_time', '09:00'),
                        '%H:%M'
                    ).time(),
                    key="start_time_setting"
                )
            
            with col2:
                end_time = st.time_input(
                    "Bitiş Saati",
                    value=datetime.strptime(
                        preferences.get('end_time', '22:00'),
                        '%H:%M'
                    ).time(),
                    key="end_time_setting"
                )
            
            # Break preferences
            st.markdown("#### 🧘‍♂️ Mola Tercihleri")
            
            col1, col2 = st.columns(2)
            
            with col1:
                break_duration = st.number_input(
                    "Mola Süresi (Dakika)",
                    min_value=5,
                    max_value=30,
                    value=preferences.get('break_duration', 15),
                    key="break_duration_setting"
                )
            
            with col2:
                study_duration = st.number_input(
                    "Çalışma Süresi (Dakika)",
                    min_value=25,
                    max_value=120,
                    value=preferences.get('study_duration', 45),
                    key="study_duration_setting"
                )
            
            submit = st.form_submit_button("Kaydet")
            
            if submit:
                preferences_update = {
                    'daily_hours': daily_hours,
                    'daily_questions': daily_questions,
                    'start_time': start_time.strftime('%H:%M'),
                    'end_time': end_time.strftime('%H:%M'),
                    'break_duration': break_duration,
                    'study_duration': study_duration
                }
                
                if self._update_study_preferences(preferences_update):
                    AlertCard.success("Çalışma tercihleri güncellendi!")
                    st.rerun()
                else:
                    AlertCard.error(
                        "Çalışma tercihleri güncellenirken bir hata oluştu!"
                    )

    def _show_notification_settings(self):
        """Display notification settings section"""
        st.markdown("### 🔔 Bildirim Ayarları")
        
        notification_settings = self._get_notification_settings()
        
        with st.form("notification_settings"):
            # Email notifications
            st.markdown("#### 📧 E-posta Bildirimleri")
            
            email_notifications = st.checkbox(
                "E-posta bildirimleri aktif",
                value=notification_settings.get('email_enabled', True),
                key="email_notifications_setting"
            )
            
            if email_notifications:
                col1, col2 = st.columns(2)
                
                with col1:
                    daily_summary = st.checkbox(
                        "Günlük özet",
                        value=notification_settings.get('daily_summary', True),
                        key="daily_summary_setting"
                    )
                    
                    achievement_notification = st.checkbox(
                        "Başarı bildirimleri",
                        value=notification_settings.get(
                            'achievement_notification', 
                            True
                        ),
                        key="achievement_notification_setting"
                    )
                
                with col2:
                    weekly_report = st.checkbox(
                        "Haftalık rapor",
                        value=notification_settings.get('weekly_report', True),
                        key="weekly_report_setting"
                    )
                    
                    study_reminder = st.checkbox(
                        "Çalışma hatırlatıcıları",
                        value=notification_settings.get('study_reminder', True),
                        key="study_reminder_setting"
                    )
            
            # Push notifications
            st.markdown("#### 🔔 Uygulama Bildirimleri")
            
            push_notifications = st.checkbox(
                "Uygulama bildirimleri aktif",
                value=notification_settings.get('push_enabled', True),
                key="push_notifications_setting"
            )
            
            if push_notifications:
                col1, col2 = st.columns(2)
                
                with col1:
                    task_notification = st.checkbox(
                        "Görev bildirimleri",
                        value=notification_settings.get('task_notification', True),
                        key="task_notification_setting"
                    )
                    
                    friend_notification = st.checkbox(
                        "Arkadaş bildirimleri",
                        value=notification_settings.get(
                            'friend_notification', 
                            True
                        ),
                        key="friend_notification_setting"
                    )
                
                with col2:
                    goal_notification = st.checkbox(
                        "Hedef bildirimleri",
                        value=notification_settings.get('goal_notification', True),
                        key="goal_notification_setting"
                    )
                    
                    forum_notification = st.checkbox(
                        "Forum bildirimleri",
                        value=notification_settings.get(
                            'forum_notification', 
                            True
                        ),
                        key="forum_notification_setting"
                    )
            
            # Quiet hours
            st.markdown("#### 🌙 Rahatsız Etme Modu")
            
            quiet_hours = st.checkbox(
                "Rahatsız etme modu aktif",
                value=notification_settings.get('quiet_hours_enabled', False),
                key="quiet_hours_setting"
            )
            
            if quiet_hours:
                col1, col2 = st.columns(2)
                
                with col1:
                    quiet_start = st.time_input(
                        "Başlangıç Saati",
                        value=datetime.strptime(
                            notification_settings.get('quiet_start', '23:00'),
                            '%H:%M'
                        ).time(),
                        key="quiet_start_setting"
                    )
                
                with col2:
                    quiet_end = st.time_input(
                        "Bitiş Saati",
                        value=datetime.strptime(
                            notification_settings.get('quiet_end', '07:00'),
                            '%H:%M'
                        ).time(),
                        key="quiet_end_setting"
                    )
            
            submit = st.form_submit_button("Kaydet")
            
            if submit:
                settings_update = {
                    'email_enabled': email_notifications,
                    'daily_summary': daily_summary if email_notifications else False,
                    'weekly_report': weekly_report if email_notifications else False,
                    'achievement_notification': achievement_notification if email_notifications else False,
                    'study_reminder': study_reminder if email_notifications else False,
                    'push_enabled': push_notifications,
                    'task_notification': task_notification if push_notifications else False,
                    'friend_notification': friend_notification if push_notifications else False,
                    'goal_notification': goal_notification if push_notifications else False,
                    'forum_notification': forum_notification if push_notifications else False,
                    'quiet_hours_enabled': quiet_hours,
                    'quiet_start': quiet_start.strftime('%H:%M') if quiet_hours else None,
                    'quiet_end': quiet_end.strftime('%H:%M') if quiet_hours else None}
                
                if self._update_notification_settings(settings_update):
                        AlertCard.success("Bildirim ayarları güncellendi!")
                        st.rerun()
                else:
                    AlertCard.error("Bildirim ayarları güncellenirken bir hata oluştu!")

    def _show_app_settings(self):
        """Display application settings section"""
        st.markdown("### 🎨 Uygulama Ayarları")
        
        app_settings = self._get_app_settings()
        
        with st.form("app_settings"):
            # Theme settings
            st.markdown("#### 🎨 Tema")
            
            theme = st.selectbox(
                "Tema Seçimi",
                ["Karanlık", "Aydınlık", "Sistem"],
                index=["dark", "light", "system"].index(
                    app_settings.get('theme', 'dark')
                ),
                key="theme_setting"
            )
            
            # Language settings
            st.markdown("#### 🌍 Dil")
            
            language = st.selectbox(
                "Uygulama Dili",
                ["Türkçe", "English"],
                index=["tr", "en"].index(
                    app_settings.get('language', 'tr')
                ),
                key="language_setting"
            )
            
            # Font size
            st.markdown("#### 📝 Yazı Boyutu")
            
            font_size = st.select_slider(
                "Yazı Boyutu",
                options=["Küçük", "Normal", "Büyük"],
                value=app_settings.get('font_size', 'Normal'),
                key="font_size_setting"
            )
            
            # Data settings
            st.markdown("#### 💾 Veri Ayarları")
            
            col1, col2 = st.columns(2)
            
            with col1:
                auto_sync = st.checkbox(
                    "Otomatik senkronizasyon",
                    value=app_settings.get('auto_sync', True),
                    key="auto_sync_setting"
                )
                
                save_offline = st.checkbox(
                    "Çevrimdışı kaydetme",
                    value=app_settings.get('save_offline', True),
                    key="save_offline_setting"
                )
            
            with col2:
                data_saver = st.checkbox(
                    "Veri tasarrufu modu",
                    value=app_settings.get('data_saver', False),
                    key="data_saver_setting"
                )
                
                analytics = st.checkbox(
                    "Analitik verisi paylaş",
                    value=app_settings.get('analytics', True),
                    key="analytics_setting"
                )
            
            # Privacy settings
            st.markdown("#### 🔒 Gizlilik")
            
            col1, col2 = st.columns(2)
            
            with col1:
                profile_visible = st.checkbox(
                    "Profili herkese açık yap",
                    value=app_settings.get('profile_visible', False),
                    key="profile_visible_setting"
                )
                
                share_progress = st.checkbox(
                    "İlerleme paylaşımı",
                    value=app_settings.get('share_progress', True),
                    key="share_progress_setting"
                )
            
            with col2:
                show_online = st.checkbox(
                    "Çevrimiçi durumu göster",
                    value=app_settings.get('show_online', True),
                    key="show_online_setting"
                )
                
                allow_messages = st.checkbox(
                    "Mesaj almaya izin ver",
                    value=app_settings.get('allow_messages', True),
                    key="allow_messages_setting"
                )
            
            submit = st.form_submit_button("Kaydet")
            
            if submit:
                settings_update = {
                    'theme': theme.lower(),
                    'language': 'tr' if language == 'Türkçe' else 'en',
                    'font_size': font_size,
                    'auto_sync': auto_sync,
                    'save_offline': save_offline,
                    'data_saver': data_saver,
                    'analytics': analytics,
                    'profile_visible': profile_visible,
                    'share_progress': share_progress,
                    'show_online': show_online,
                    'allow_messages': allow_messages
                }
                
                if self._update_app_settings(settings_update):
                    AlertCard.success("Uygulama ayarları güncellendi!")
                    st.rerun()
                else:
                    AlertCard.error("Uygulama ayarları güncellenirken bir hata oluştu!")

    def _get_profile_data(self) -> Dict[str, Any]:
        """Get user profile data from database"""
        try:
            profile_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT name, email, city, grade, study_type,
                       target_university, target_department, target_rank
                FROM users
                WHERE id = ?
            ''', (self.user_id,), fetch_all=False))
            
            return self._row_to_dict(profile_query)
        except DatabaseError as e:
            st.error(f"Error fetching profile data: {str(e)}")
            return {}

    def _get_study_preferences(self) -> Dict[str, Any]:
        """Get user study preferences from database"""
        try:
            prefs_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT * FROM study_preferences
                WHERE user_id = ?
            ''', (self.user_id,), fetch_all=False))
            
            return self._row_to_dict(prefs_query)
        except DatabaseError as e:
            st.error(f"Error fetching study preferences: {str(e)}")
            return {}

    def _get_notification_settings(self) -> Dict[str, Any]:
        """Get user notification settings from database"""
        try:
            notif_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT * FROM notification_settings
                WHERE user_id = ?
            ''', (self.user_id,), fetch_all=False))
            
            return self._row_to_dict(notif_query)
        except DatabaseError as e:
            st.error(f"Error fetching notification settings: {str(e)}")
            return {}

    def _get_app_settings(self) -> Dict[str, Any]:
        """Get user application settings from database"""
        try:
            settings_query = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT * FROM app_settings
                WHERE user_id = ?
            ''', (self.user_id,), fetch_all=False))
            
            return self._row_to_dict(settings_query)
        except DatabaseError as e:
            st.error(f"Error fetching app settings: {str(e)}")
            return {}

    def _update_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Update user profile in database"""
        try:
            with db_transaction() as conn:
                self.db_manager.execute_query('''
                    UPDATE users
                    SET name = ?, email = ?, city = ?, grade = ?,
                        study_type = ?, target_university = ?,
                        target_department = ?, target_rank = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    profile_data['name'],
                    profile_data['email'],
                    profile_data['city'],
                    profile_data['grade'],
                    profile_data['study_type'],
                    profile_data['target_university'],
                    profile_data['target_department'],
                    profile_data['target_rank'],
                    self.user_id
                ), fetch_all=False)
                
                return True
        except DatabaseError as e:
            st.error(f"Error updating profile: {str(e)}")
            return False

    def _update_password(self, new_password: str) -> bool:
        """Update user password in database"""
        try:
            with db_transaction() as conn:
                hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                
                self.db_manager.execute_query('''
                    UPDATE users
                    SET password = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (hashed_password, self.user_id), fetch_all=False)
                
                return True
        except DatabaseError as e:
            st.error(f"Error updating password: {str(e)}")
            return False

    def _verify_password(self, password: str) -> bool:
        """Verify user's current password"""
        try:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            result = cast(Optional[Row], self.db_manager.execute_query('''
                SELECT 1 FROM users
                WHERE id = ? AND password = ?
            ''', (self.user_id, hashed_password), fetch_all=False))
            
            return bool(result)
        except DatabaseError as e:
            st.error(f"Error verifying password: {str(e)}")
            return False

    def _update_study_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update study preferences in database"""
        try:
            with db_transaction() as conn:
                self.db_manager.execute_query('''
                    INSERT OR REPLACE INTO study_preferences
                    (user_id, daily_hours, daily_questions, start_time,
                     end_time, break_duration, study_duration, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    self.user_id,
                    preferences['daily_hours'],
                    preferences['daily_questions'],
                    preferences['start_time'],
                    preferences['end_time'],
                    preferences['break_duration'],
                    preferences['study_duration']
                ), fetch_all=False)
                
                return True
        except DatabaseError as e:
            st.error(f"Error updating study preferences: {str(e)}")
            return False

    def _update_notification_settings(self, settings: Dict[str, Any]) -> bool:
        """Update notification settings in database"""
        try:
            with db_transaction() as conn:
                self.db_manager.execute_query('''
                    INSERT OR REPLACE INTO notification_settings
                    (user_id, email_enabled, push_enabled, quiet_hours_enabled,
                     quiet_start, quiet_end, daily_summary, weekly_report,
                     achievement_notification, study_reminder, task_notification,
                     friend_notification, goal_notification, forum_notification)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.user_id,
                    settings['email_enabled'],
                    settings['push_enabled'],
                    settings['quiet_hours_enabled'],
                    settings['quiet_start'],
                    settings['quiet_end'],
                    settings['daily_summary'],
                    settings['weekly_report'],
                    settings['achievement_notification'],
                    settings['study_reminder'],
                    settings['task_notification'],
                    settings['friend_notification'],
                    settings['goal_notification'],
                    settings['forum_notification']
                ), fetch_all=False)
                
                return True
        except DatabaseError as e:
            st.error(f"Error updating notification settings: {str(e)}")
            return False

    def _update_app_settings(self, settings: Dict[str, Any]) -> bool:
        """Update application settings in database"""
        try:
            with db_transaction() as conn:
                self.db_manager.execute_query('''
                    INSERT OR REPLACE INTO app_settings
                    (user_id, theme, language, font_size, auto_sync,
                     save_offline, data_saver, analytics, profile_visible,
                     share_progress, show_online, allow_messages)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.user_id,
                    settings['theme'],
                    settings['language'],
                    settings['font_size'],
                    settings['auto_sync'],
                    settings['save_offline'],
                    settings['data_saver'],
                    settings['analytics'],
                    settings['profile_visible'],
                    settings['share_progress'],
                    settings['show_online'],
                    settings['allow_messages']
                ), fetch_all=False)
                
                return True
        except DatabaseError as e:
            st.error(f"Error updating app settings: {str(e)}")
            return False

    def _get_grade_index(self, grade: str) -> int:
        """Get index for grade selectbox"""
        grades = ["9. Sınıf", "10. Sınıf", "11. Sınıf", "12. Sınıf", "Mezun"]
        try:
            return grades.index(grade)
        except ValueError:
            return 0

    def _get_study_type_index(self, study_type: str) -> int:
        """Get index for study type selectbox"""
        types = ["Sayısal", "Eşit Ağırlık", "Sözel", "Dil"]
        try:
            return types.index(study_type)
        except ValueError:
            return 0

def show_settings():
    """Main function to show settings page"""
    user_id = st.session_state.get('user_id')
    if user_id:
        settings_page = SettingsPage(user_id=user_id)
        settings_page.show()
    else:
        st.error("Lütfen giriş yapın.")

if __name__ == "__main__":
    show_settings()     
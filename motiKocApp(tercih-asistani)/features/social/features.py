# features/social/features.py 
import streamlit as st
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import sqlite3  # Ensure sqlite3 is imported


class SocialFeatures:
    def __init__(self, user_id: int, db_conn: sqlite3.Connection):
        self.user_id = user_id
        self.db_conn = db_conn

        # Ensure that the connection returns rows as sqlite3.Row
        self.db_conn.row_factory = sqlite3.Row

    def show(self):
        st.title("👥 Sosyal Özellikler")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Çalışma Grupları",
            "Arkadaşlar",
            "Başarılar",
            "Liderlik Tablosu"
        ])
        
        with tab1:
            self._show_study_groups()
        
        with tab2:
            self._show_friends()
        
        with tab3:
            self._show_achievements()
        
        with tab4:
            self._show_leaderboard()

    # -----------------------------
    # Study Groups Methods
    # -----------------------------

    def _show_study_groups(self):
        """Display study groups features"""
        st.markdown("### 👥 Çalışma Grupları")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show existing groups
            my_groups = self._get_my_groups()
            
            if my_groups:
                for group in my_groups:
                    with st.expander(f"📚 {group['name']}"):
                        st.markdown(f"**Tür:** {group['group_type']}")
                        st.markdown(f"**Açıklama:** {group['description']}")
                        st.markdown(f"**Üye Sayısı:** {group['member_count']}/{group['max_members']}")
                        
                        # Show members
                        st.markdown("**Üyeler:**")
                        for member in group['members']:
                            st.markdown(f"- {member['username']} ({member['role']})")
                        
                        # Show actions based on role
                        if group['is_admin']:
                            if st.button("Grubu Düzenle", key=f"edit_group_{group['id']}"):
                                self._show_edit_group_form(group)
                            
                            if st.button("Grubu Sil", key=f"delete_group_{group['id']}"):
                                if self._delete_group(group['id']):
                                    st.success("Grup silindi!")
                                    st.rerun()
                        else:
                            if st.button("Gruptan Ayrıl", key=f"leave_group_{group['id']}"):
                                if self._leave_group(group['id']):
                                    st.success("Gruptan ayrıldınız!")
                                    st.rerun()
            else:
                st.info("Henüz bir çalışma grubuna üye değilsiniz.")
        
        with col2:
            # Create new group button
            if st.button("➕ Yeni Grup Oluştur"):
                self._show_create_group_form()
            
            # Search groups
            st.markdown("### 🔍 Grup Ara")
            search_query = st.text_input("Grup Adı")
            
            if search_query:
                groups = self._search_groups(search_query)
                
                if groups:
                    for group in groups:
                        st.markdown(f"""
                        <div style='padding: 10px; 
                                  background-color: rgba(255, 255, 255, 0.05);
                                  border-radius: 5px;
                                  margin: 5px 0;'>
                            <h4>{group['name']}</h4>
                            <p>{group['description']}</p>
                            <small>{group['member_count']}/{group['max_members']} üye</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if group['can_join']:
                            if st.button("Katıl", key=f"join_group_{group['id']}"):
                                if self._join_group(group['id']):
                                    st.success("Gruba katıldınız!")
                                    st.rerun()
                else:
                    st.info("Sonuç bulunamadı.")

    def _get_my_groups(self) -> List[sqlite3.Row]:
        """Get user's study groups"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT 
                    g.*,
                    COUNT(DISTINCT m.user_id) as member_count,
                    m2.role as user_role
                FROM study_groups g
                JOIN group_members m ON g.id = m.group_id
                JOIN group_members m2 ON g.id = m2.group_id AND m2.user_id = ?
                GROUP BY g.id
                ORDER BY g.name
            ''', (self.user_id,))
            
            groups = cursor.fetchall()
            
            # Get members for each group
            for group in groups:
                cursor.execute('''
                    SELECT u.username, m.role
                    FROM group_members m
                    JOIN users u ON m.user_id = u.id
                    WHERE m.group_id = ?
                    ORDER BY m.role DESC, u.username
                ''', (group['id'],))
                
                group['members'] = cursor.fetchall()
                group['is_admin'] = group['user_role'] == 'admin'
            
            return groups
        except Exception as e:
            print(f"Error getting groups: {str(e)}")
            return []

    def _search_groups(self, query: str) -> List[sqlite3.Row]:
        """Search study groups"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
               SELECT 
                   g.*,
                   COUNT(DISTINCT m.user_id) as member_count,
                   NOT EXISTS (
                       SELECT 1 FROM group_members
                       WHERE group_id = g.id AND user_id = ?
                   ) as can_join
               FROM study_groups g
               LEFT JOIN group_members m ON g.id = m.group_id
               WHERE g.name LIKE ? OR g.description LIKE ?
               GROUP BY g.id
               HAVING member_count < g.max_members
               ORDER BY g.name
           ''', (self.user_id, f"%{query}%", f"%{query}%"))
           
            return cursor.fetchall()
        except Exception as e:
           print(f"Error searching groups: {str(e)}")
           return []

    def _join_group(self, group_id: int) -> bool:
       """Join study group"""
       try:
           cursor = self.db_conn.cursor()
           cursor.execute('''
               INSERT INTO group_members (group_id, user_id, role)
               VALUES (?, ?, 'member')
           ''', (group_id, self.user_id))
           
           self.db_conn.commit()
           return True
       except Exception as e:
           print(f"Error joining group: {str(e)}")
           return False

    def _leave_group(self, group_id: int) -> bool:
        """Leave study group"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                DELETE FROM group_members
                WHERE group_id = ? AND user_id = ?
            ''', (group_id, self.user_id))
            
            self.db_conn.commit()
            return True
        except Exception as e:
            print(f"Error leaving group: {str(e)}")
            return False

    def _delete_group(self, group_id: int) -> bool:
        """Delete study group"""
        try:
            cursor = self.db_conn.cursor()
            
            # Delete members first
            cursor.execute('''
                DELETE FROM group_members
                WHERE group_id = ?
            ''', (group_id,))
            
            # Delete group
            cursor.execute('''
                DELETE FROM study_groups
                WHERE id = ? AND creator_id = ?
            ''', (group_id, self.user_id))
            
            self.db_conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting group: {str(e)}")
            return False

    def _show_edit_group_form(self, group: sqlite3.Row):
        """Display form to edit a study group"""
        st.subheader("Grubu Düzenle")
        with st.form(f"edit_group_{group['id']}"):
            name = st.text_input("Grup Adı", value=group['name']) or group['name']
            group_type = st.selectbox("Grup Türü", ["Type1", "Type2"], index=0)  # Replace with actual types
            description = st.text_area("Açıklama", value=group['description']) or group['description']
            max_members = st.number_input("Maksimum Üye Sayısı", min_value=1, value=group['max_members'])
            submit = st.form_submit_button("Güncelle")
            
            if submit:
                # Ensure that name and description are not None
                name = name.strip()
                description = description.strip()
                
                if not name:
                    st.error("Grup adı boş olamaz.")
                else:
                    # Update group in the database
                    success = self._update_group(group['id'], name, group_type, description, max_members)
                    if success:
                        st.success("Grup başarıyla güncellendi!")
                        st.rerun()
                    else:
                        st.error("Grup güncellenirken bir hata oluştu.")

    def _update_group(self, group_id: int, name: str, group_type: str, description: str, max_members: int) -> bool:
        """Update group details in the database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                UPDATE study_groups
                SET name = ?, group_type = ?, description = ?, max_members = ?
                WHERE id = ? AND creator_id = ?
            ''', (name, group_type, description, max_members, group_id, self.user_id))
            self.db_conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating group: {str(e)}")
            return False

    def _show_create_group_form(self):
        """Display form to create a new study group"""
        st.subheader("Yeni Grup Oluştur")
        with st.form("create_group"):
            name = st.text_input("Grup Adı") or ""
            group_type = st.selectbox("Grup Türü", ["Type1", "Type2"], index=0)  # Replace with actual types
            description = st.text_area("Açıklama") or ""
            max_members = st.number_input("Maksimum Üye Sayısı", min_value=1, value=10)
            submit = st.form_submit_button("Oluştur")
            
            if submit:
                name = name.strip()
                description = description.strip()
                
                if not name:
                    st.error("Grup adı boş olamaz.")
                else:
                    # Create group in the database
                    success = self._create_group(name, group_type, description, max_members)
                    if success:
                        st.success("Grup başarıyla oluşturuldu!")
                        st.rerun()
                    else:
                        st.error("Grup oluşturulurken bir hata oluştu.")

    def _create_group(self, name: str, group_type: str, description: str, max_members: int) -> bool:
        """Create a new study group in the database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO study_groups (name, group_type, description, max_members, creator_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, group_type, description, max_members, self.user_id))
            
            # Get the newly created group ID
            group_id = cursor.lastrowid
            
            # Add the creator as an admin member
            cursor.execute('''
                INSERT INTO group_members (group_id, user_id, role)
                VALUES (?, ?, 'admin')
            ''', (group_id, self.user_id))
            
            self.db_conn.commit()
            return True
        except Exception as e:
            print(f"Error creating group: {str(e)}")
            return False

    # -----------------------------
    # Friends Methods
    # -----------------------------

    def _show_friends(self):
        """Display friends features"""
        st.markdown("### 👥 Arkadaşlar")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show friends list
            friends = self._get_friends()
            
            if friends:
                for friend in friends:
                    with st.container():
                        col_a, col_b, col_c = st.columns([3, 1, 1])
                        
                        with col_a:
                            st.markdown(f"""
                            <div style='padding: 10px; 
                                      background-color: rgba(255, 255, 255, 0.05);
                                      border-radius: 5px;'>
                                <h4>{friend['username']}</h4>
                                <p>{friend['study_type']} | {friend['grade']}</p>
                                <small>Son aktif: {friend['last_active']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b:
                            if st.button("Profil", key=f"view_profile_{friend['id']}"):
                                self._show_friend_profile(friend['id'])
                        
                        with col_c:
                            if st.button("Çıkar", key=f"remove_friend_{friend['id']}"):
                                if self._remove_friend(friend['id']):
                                    st.success("Arkadaş çıkarıldı!")
                                    st.rerun()
            else:
                st.info("Henüz arkadaşınız yok.")
        
        with col2:
            # Add friend
            st.markdown("### 👋 Arkadaş Ekle")
            with st.form("add_friend_form"):
                username = st.text_input("Kullanıcı Adı") or ""
                submit = st.form_submit_button("Ekle")
                
                if submit and username.strip():
                    username = username.strip()
                    result = self._add_friend(username)
                    if result['success']:
                        st.success(result['message'])
                        st.rerun()
                    else:
                        st.error(result['message'])
                elif submit:
                    st.error("Kullanıcı adı boş olamaz.")
            
            # Friend requests
            st.markdown("### 📨 Arkadaşlık İstekleri")
            requests = self._get_friend_requests()
            
            if requests:
                for request in requests:
                    st.markdown(f"""
                    <div style='padding: 10px; 
                              background-color: rgba(255, 255, 255, 0.05);
                              border-radius: 5px;
                              margin: 5px 0;'>
                        <p>{request['username']}</p>
                        <small>{request['created_at']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        if st.button("Kabul Et", key=f"accept_request_{request['id']}"):
                            if self._handle_friend_request(request['id'], True):
                                st.success("İstek kabul edildi!")
                                st.rerun()
                    
                    with col_y:
                        if st.button("Reddet", key=f"reject_request_{request['id']}"):
                            if self._handle_friend_request(request['id'], False):
                                st.success("İstek reddedildi!")
                                st.rerun()
            else:
                st.info("Bekleyen istek yok.")

    def _get_friends(self) -> List[sqlite3.Row]:
        """Get user's friends"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT 
                    u.id, u.username, u.grade, u.study_type,
                    ul.current_level,
                    COALESCE(MAX(s.date), 'Hiç aktif değil') as last_active
                FROM users u
                JOIN friendships f ON 
                    (f.user_id = ? AND f.friend_id = u.id) OR
                    (f.friend_id = ? AND f.user_id = u.id)
                JOIN user_levels ul ON u.id = ul.user_id
                LEFT JOIN study_logs s ON u.id = s.user_id
                WHERE f.status = 'accepted'
                GROUP BY u.id
                ORDER BY u.username
            ''', (self.user_id, self.user_id))
            
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting friends: {str(e)}")
            return []

    def _add_friend(self, username: str) -> Dict[str, Any]:
        """Send friend request"""
        try:
            cursor = self.db_conn.cursor()
            
            # Check if user exists
            cursor.execute('''
                SELECT id FROM users
                WHERE username = ?
            ''', (username,))
            
            user = cursor.fetchone()
            if not user:
                return {
                    'success': False,
                    'message': 'Kullanıcı bulunamadı.'
                }
            
            friend_id = user['id']
            
            if friend_id == self.user_id:
                return {
                    'success': False,
                    'message': 'Kendinize arkadaşlık isteği gönderemezsiniz.'
                }
            
            # Check if already friends or request pending
            cursor.execute('''
                SELECT status FROM friendships
                WHERE (user_id = ? AND friend_id = ?) OR
                      (user_id = ? AND friend_id = ?)
            ''', (self.user_id, friend_id, friend_id, self.user_id))
            
            friendship = cursor.fetchone()
            if friendship:
                if friendship['status'] == 'pending':
                    return {
                        'success': False,
                        'message': 'Arkadaşlık isteği zaten gönderilmiş.'
                    }
                elif friendship['status'] == 'accepted':
                    return {
                        'success': False,
                        'message': 'Bu kullanıcı zaten arkadaşınız.'
                    }
            
            # Send request
            cursor.execute('''
                INSERT INTO friendships (user_id, friend_id, status, created_at)
                VALUES (?, ?, 'pending', ?)
            ''', (self.user_id, friend_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            self.db_conn.commit()
            
            return {
                'success': True,
                'message': 'Arkadaşlık isteği gönderildi.'
            }
            
        except Exception as e:
            print(f"Error adding friend: {str(e)}")
            return {
                'success': False,
                'message': 'Bir hata oluştu.'
            }

    def _get_friend_requests(self) -> List[sqlite3.Row]:
        """Get pending friend requests"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT f.id, u.username, f.created_at
                FROM friendships f
                JOIN users u ON f.user_id = u.id
                WHERE f.friend_id = ? AND f.status = 'pending'
                ORDER BY f.created_at DESC
            ''', (self.user_id,))
            
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting friend requests: {str(e)}")
            return []

    def _handle_friend_request(self, request_id: int, accept: bool) -> bool:
        """Handle friend request"""
        try:
            cursor = self.db_conn.cursor()
            
            if accept:
                cursor.execute('''
                    UPDATE friendships
                    SET status = 'accepted'
                    WHERE id = ? AND friend_id = ?
                ''', (request_id, self.user_id))
            else:
                cursor.execute('''
                    DELETE FROM friendships
                    WHERE id = ? AND friend_id = ?
                ''', (request_id, self.user_id))
            
            self.db_conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error handling friend request: {str(e)}")
            return False

    def _remove_friend(self, friend_id: int) -> bool:
        """Remove friend"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                DELETE FROM friendships
                WHERE (user_id = ? AND friend_id = ?) OR
                      (user_id = ? AND friend_id = ?)
            ''', (self.user_id, friend_id, friend_id, self.user_id))
            
            self.db_conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing friend: {str(e)}")
            return False

    def _show_friend_profile(self, friend_id: int):
        """Display a friend's profile"""
        st.subheader("Arkadaş Profili")
        friend = self._get_user_profile(friend_id)
        if friend:
            st.write(f"**Kullanıcı Adı:** {friend['username']}")
            st.write(f"**Sınıf:** {friend['grade']}")
            st.write(f"**Çalışma Türü:** {friend['study_type']}")
            st.write(f"**Seviye:** {friend['current_level']}")
            # Add more profile details as needed
        else:
            st.error("Arkadaş profili bulunamadı.")

    def _get_user_profile(self, user_id: int) -> Optional[sqlite3.Row]:
        """Fetch a user's profile information"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT username, grade, study_type, current_level
                FROM users
                JOIN user_levels ON users.id = user_levels.user_id
                WHERE users.id = ?
            ''', (user_id,))
            return cursor.fetchone()
        except Exception as e:
            print(f"Error fetching user profile: {str(e)}")
            return None

    # -----------------------------
    # Achievements Methods
    # -----------------------------

    def _show_achievements(self):
        """Display achievements and badges"""
        st.markdown("### 🏆 Başarılar")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show achievements
            achievements = self._get_achievements()
            
            if achievements:
                for achievement in achievements:
                    st.markdown(f"""
                    <div style='padding: 15px; 
                              background-color: rgba(255, 255, 255, 0.05);
                              border-radius: 10px;
                              margin: 10px 0;'>
                        <h4>🏅 {achievement['title']}</h4>
                        <p>{achievement['description']}</p>
                        <small>{achievement['date']}</small>
                        {f'<br><small>🎉 {achievement["likes"]} beğeni</small>' 
                         if achievement["likes"] > 0 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if not achievement['is_own']:
                        like_button_label = "👍 Beğen" if not achievement['liked'] else "❤️ Beğenildi"
                        if st.button(
                            like_button_label,
                            key=f"like_achievement_{achievement['id']}"
                        ):
                            if self._toggle_achievement_like(achievement['id']):
                                st.success("Beğeni güncellendi!")
                                st.rerun()
            else:
                st.info("Henüz başarı kaydınız yok.")
        
        with col2:
            # Show badges
            st.markdown("### 🌟 Rozetler")
            badges = self._get_badges()
            
            if badges:
                for category, category_badges in badges.items():
                    st.markdown(f"#### {category}")
                    
                    for badge in category_badges:
                        st.markdown(f"""
                        <div style='padding: 10px; 
                                  background-color: rgba(255, 255, 255, 0.05);
                                  border-radius: 5px;
                                  margin: 5px 0;'>
                            <h4>🌟 {badge['name']}</h4>
                            <p>{badge['description']}</p>
                            <small>Kazanıldı: {badge['earned_date']}</small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Henüz rozet kazanmadınız.")

    def _get_achievements(self) -> List[sqlite3.Row]:
        """Get achievements feed"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT 
                    a.*,
                    u.username,
                    COUNT(al.id) as likes,
                    EXISTS (
                        SELECT 1 FROM achievement_likes
                        WHERE achievement_id = a.id AND user_id = ?
                    ) as liked,
                    CASE WHEN a.user_id = ? THEN 1 ELSE 0 END as is_own
                FROM achievements a
                JOIN users u ON a.user_id = u.id
                LEFT JOIN achievement_likes al ON a.id = al.achievement_id
                WHERE a.user_id = ? OR
                      a.user_id IN (
                        SELECT CASE
                            WHEN f.user_id = ? THEN f.friend_id
                            ELSE f.user_id
                        END
                        FROM friendships f
                        WHERE (f.user_id = ? OR f.friend_id = ?)
                        AND f.status = 'accepted'
                      )
                GROUP BY a.id
                ORDER BY a.date DESC
                LIMIT 50
            ''', (
                self.user_id, self.user_id, 
                self.user_id, self.user_id,
                self.user_id, self.user_id
            ))
            
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting achievements: {str(e)}")
            return []

    def _toggle_achievement_like(self, achievement_id: int) -> bool:
        """Toggle achievement like"""
        try:
            cursor = self.db_conn.cursor()
            
            # Check if already liked
            cursor.execute('''
                SELECT id FROM achievement_likes
                WHERE achievement_id = ? AND user_id = ?
            ''', (achievement_id, self.user_id))
            
            like = cursor.fetchone()
            
            if like:
                # Unlike
                cursor.execute('''
                    DELETE FROM achievement_likes
                    WHERE id = ?
                ''', (like['id'],))
            else:
                # Like
                cursor.execute('''
                    INSERT INTO achievement_likes (achievement_id, user_id)
                    VALUES (?, ?)
                ''', (achievement_id, self.user_id))
            
            self.db_conn.commit()
            return True
        except Exception as e:
            print(f"Error toggling achievement like: {str(e)}")
            return False

    def _get_badges(self) -> Dict[str, List[sqlite3.Row]]:
        """Get user's badges"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT 
                    b.*,
                    ub.earned_date
                FROM badges b
                JOIN user_badges ub ON b.id = ub.badge_id
                WHERE ub.user_id = ?
                ORDER BY b.category, ub.earned_date DESC
            ''', (self.user_id,))
            
            badges = cursor.fetchall()
            
            # Group by category
            categorized: Dict[str, List[sqlite3.Row]] = {}
            for badge in badges:
                category = badge['category']
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(badge)
            
            return categorized
        except Exception as e:
            print(f"Error getting badges: {str(e)}")
            return {}

    # -----------------------------
    # Leaderboard Methods
    # -----------------------------

    def _show_leaderboard(self):
        """Display leaderboard"""
        st.markdown("### 🏆 Liderlik Tablosu")
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Zaman Aralığı",
            ["Haftalık", "Aylık", "Tüm Zamanlar"]
        )
        
        # Category selection
        category = st.selectbox(
            "Kategori",
            ["Çalışma Süresi", "XP", "Soru Çözümü"]
        )
        
        # Get leaderboard data
        leaderboard = self._get_leaderboard_data(timeframe, category)
        
        if leaderboard:
            # Display top 3
            col1, col2, col3 = st.columns([1, 1.5, 1])
            
            if len(leaderboard) >= 2:
                with col1:
                    self._show_leaderboard_card(
                        leaderboard[1], 
                        2, 
                        "🥈",
                        category
                    )
            
            if len(leaderboard) >= 1:
                with col2:
                    self._show_leaderboard_card(
                        leaderboard[0], 
                        1, 
                        "🏆",
                        category
                    )
            
            if len(leaderboard) >= 3:
                with col3:
                    self._show_leaderboard_card(
                        leaderboard[2], 
                        3, 
                        "🥉",
                        category
                    )
            
            # Display rest of the leaderboard
            if len(leaderboard) > 3:
                st.markdown("---")
                for i, entry in enumerate(leaderboard[3:], 4):
                    st.markdown(f"""
                    <div style='padding: 10px; 
                              background-color: rgba(255, 255, 255, 0.05);
                              border-radius: 5px;
                              margin: 5px 0;
                              display: flex;
                              justify-content: space-between;
                              align-items: center;'>
                        <div>
                            <strong>#{i}</strong> {entry['username']}
                        </div>
                        <div>
                            {self._format_leaderboard_value(entry['value'], category)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Henüz sıralama verisi bulunmuyor.")

    def _get_leaderboard_data(self, timeframe: str, category: str) -> List[sqlite3.Row]:
        """Get leaderboard data"""
        try:
            cursor = self.db_conn.cursor()
           
            # Set date filter based on timeframe
            date_filter = ""
            if timeframe == "Haftalık":
                date_filter = "AND date >= date('now', '-7 days')"
            elif timeframe == "Aylık":
                date_filter = "AND date >= date('now', '-30 days')"
           
            # Get data based on category
            if category == "Çalışma Süresi":
                cursor.execute(f'''
                    SELECT 
                        u.id,
                        u.username,
                        COALESCE(SUM(s.duration), 0) / 60.0 as value
                    FROM users u
                    LEFT JOIN study_logs s ON 
                        u.id = s.user_id {date_filter}
                    GROUP BY u.id
                    HAVING value > 0
                    ORDER BY value DESC
                    LIMIT 20
                ''')
            elif category == "XP":
                cursor.execute(f'''
                    SELECT 
                        u.id,
                        u.username,
                        ul.total_xp as value
                    FROM users u
                    JOIN user_levels ul ON u.id = ul.user_id
                    WHERE ul.total_xp > 0
                    ORDER BY value DESC
                    LIMIT 20
                ''')
            else:  # Soru Çözümü
                cursor.execute(f'''
                    SELECT 
                        u.id,
                        u.username,
                        COALESCE(SUM(
                            CASE 
                                WHEN m.exam_type = 'TYT' THEN 
                                    json_extract(m.subject_results, '$.correct') * 1.0
                                ELSE 
                                    json_extract(m.subject_results, '$.correct') * 1.5
                            END
                        ), 0) as value
                    FROM users u
                    LEFT JOIN mock_exams m ON 
                        u.id = m.user_id {date_filter}
                    GROUP BY u.id
                    HAVING value > 0
                    ORDER BY value DESC
                    LIMIT 20
                ''')
           
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting leaderboard data: {str(e)}")
            return []

    def _show_leaderboard_card(self, 
                             entry: sqlite3.Row, 
                             position: int,
                             medal: str,
                             category: str):
        """Display leaderboard position card"""
        st.markdown(f"""
        <div style='
            padding: 15px;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            text-align: center;
            margin: {"20px 0 0 0" if position == 1 else "40px 0 0 0"}
        '>
            <h2>{medal}</h2>
            <h3>#{position}</h3>
            <h4>{entry['username']}</h4>
            <p>{self._format_leaderboard_value(entry['value'], category)}</p>
        </div>
        """, unsafe_allow_html=True)

    def _format_leaderboard_value(self, value: float, category: str) -> str:
        """Format leaderboard value based on category"""
        if category == "Çalışma Süresi":
            hours = value  # Assuming value is already in hours
            return f"{hours:.1f} saat"
        elif category == "XP":
            return f"{int(value):,} XP"
        else:  # Soru Çözümü
            return f"{int(value):,} soru"
def show_social_features():
    """
    Initialize and display the SocialFeatures component.
    """
    if 'db_conn' not in st.session_state:
        st.error("Veritabanı bağlantısı bulunamadı.")
        return

    user_id = st.session_state.get('user_id')
    if user_id is None:
        st.error("Kullanıcı kimliği bulunamadı.")
        return

    social = SocialFeatures(user_id=user_id, db_conn=st.session_state.db_conn)
    social.show()
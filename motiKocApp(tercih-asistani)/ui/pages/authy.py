# ui/pages/authy.py

from core.auth import register_user, login_user, AuthError
import streamlit as st

def show_login_page():
    st.title("🎯 MotiKoç'a Hoş Geldiniz")
    
    tab1, tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı")
            password = st.text_input("Şifre", type="password")
            submit = st.form_submit_button("Giriş Yap")
            
            if submit:
                if username and password:
                    try:
                        user_id = login_user(username, password)
                        if user_id:
                            st.session_state.user_id = user_id
                            st.session_state.username = username
                            st.success("Giriş başarılı!")
                            st.rerun()
                        else:
                            st.error("Hatalı kullanıcı adı veya şifre!")
                    except AuthError as e:
                        st.error(str(e))
                else:
                    st.error("Lütfen tüm alanları doldurun!")
    
    with tab2:
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                reg_username = st.text_input("Kullanıcı Adı*", key="reg_username")
                reg_password = st.text_input("Şifre*", type="password", key="reg_password")
                reg_name = st.text_input("Ad Soyad*", key="reg_name")
                reg_email = st.text_input("E-posta*", key="reg_email")
                reg_city = st.text_input("Şehir", key="reg_city")
            
            with col2:
                reg_grade = st.selectbox(
                    "Sınıf Düzeyi*",
                    ["9. Sınıf", "10. Sınıf", "11. Sınıf", "12. Sınıf", "Mezun"],
                    key="reg_grade"
                )
                reg_study_type = st.selectbox(
                    "Alan*",
                    ["Sayısal", "Eşit Ağırlık", "Sözel", "Dil"],
                    key="reg_study_type"
                )
                reg_target_university = st.text_input("Hedef Üniversite", key="reg_target_university")
                reg_target_department = st.text_input("Hedef Bölüm", key="reg_target_department")
                reg_target_rank = st.number_input("Hedef Sıralama", min_value=1, step=1, key="reg_target_rank")
            
            submit_register = st.form_submit_button("Kayıt Ol")
            
            if submit_register:
                if reg_username and reg_password and reg_name and reg_email and reg_grade and reg_study_type:
                    user_data = {
                        'username': reg_username,
                        'password': reg_password,
                        'name': reg_name,
                        'email': reg_email,
                        'grade': reg_grade,
                        'city': reg_city,
                        'target_university': reg_target_university,
                        'target_department': reg_target_department,
                        'target_rank': reg_target_rank,
                        'study_type': reg_study_type
                    }
                    try:
                        user_id = register_user(user_data)
                        if user_id:
                            st.success("Kayıt başarılı! Giriş yapabilirsiniz.")
                        else:
                            st.error("Kayıt işlemi sırasında bir sorun oluştu.")
                    except AuthError as e:
                        st.error(str(e))
                else:
                    st.error("Lütfen zorunlu alanları doldurun!")

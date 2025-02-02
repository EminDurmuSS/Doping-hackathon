import streamlit as st
import pandas as pd
import openai, pinecone, time, os, ast
from tqdm import tqdm  # Optional progress display

# =============================================================================
# 1) APP CONFIGURATION & CUSTOM STYLE
# =============================================================================
st.set_page_config(page_title="Soru Çözüm Öneri Sistemi", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a cleaner look
custom_css = """
<style>
    .main {background-color: #F5F5F5; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2c3e50, #2c3e50); color: white; }
    h1 {color: #2c3e50;}
    h2 {color: #34495e;}
    .stButton>button {background-color: #2c3e50; color: white; border-radius: 5px;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("Soru Çözüm Öneri Sistemi")
st.markdown(
    """
    **Hoş geldiniz!**  
    Bu sistem, cevaplayamadığınız bir soruya benzer soruları önererek konuyu daha iyi anlamanıza yardımcı olur.
    
    **Nasıl Kullanılır?**
    - **Sol panelde:** Sorular arasında arama yapabilir ve bir soru seçebilirsiniz.
    - **Ana panelde:** Seçtiğiniz soruyu görüntüleyip, cevap seçeneklerinden doğru cevabı işaretleyebilirsiniz.
    
    Ayrıca, eğer cevap yanlış ise, önerilen benzer sorulardan istediğinize tıklayarak, o soruya hızlıca geçebilirsiniz.
    
    İyi çalışmalar!
    """
)

# =============================================================================
# 2) SIDEBAR AYARLARI & API KULLANIMI
# =============================================================================
st.sidebar.header("Ayarlar & Bilgiler")
debug_mode = st.sidebar.checkbox("Debug Modunu Etkinleştir", value=False)

# API Keys (replace with your own or use Streamlit secrets)
openai.api_key = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# Pinecone & Embedding Settings
PINECONE_ENV = "us-east-1"
INDEX_NAME = "sorular-index"
BATCH_SIZE = 50
EMBEDDING_MODEL = "text-embedding-3-large"
EMBED_DIMENSIONS = 3072

# CSV Data file (update the path as needed)
CSV_PATH = "sorular_cozumleri_featureslerle.csv"

# =============================================================================
# 3) PINECONE INDEX BAĞLANTISI
# =============================================================================
st.sidebar.info("Pinecone indeksine bağlanılıyor...")
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    st.sidebar.info(f"Index '{INDEX_NAME}' bulunamadı. Oluşturuluyor...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
else:
    st.sidebar.success(f"Index '{INDEX_NAME}' zaten mevcut.")

index = pc.Index(INDEX_NAME)

# =============================================================================
# 4) CSV VERİLERİNİN YÜKLENMESİ VE ÖN İŞLEM
# =============================================================================
@st.cache_data(show_spinner=True)
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error("CSV dosyası yüklenirken hata oluştu!")
        if debug_mode:
            st.exception(e)
        return pd.DataFrame()
    
    # Combine key text fields, including extra MEB/müfredat fields for richer context
    def combine_text(row):
        parts = []
        for field in ["soru_metni", "çözüm", "alt_konular", "konu", "matematik_formulu"]:
            if pd.notnull(row.get(field)):
                parts.append(str(row[field]))
        for field in ["meb_kazanım", "taxonomy", "soru_türü", "difficulty", "sinif", "sik_yapilan_hatalar"]:
            if pd.notnull(row.get(field)):
                parts.append(str(row[field]))
        return " ".join(parts)
    
    df["combined_text"] = df.apply(combine_text, axis=1)
    return df

df = load_data(CSV_PATH)
if df.empty:
    st.error("Veriler yüklenemedi. Lütfen CSV dosyanızı kontrol edin.")
else:
    st.sidebar.success(f"Toplam {len(df)} soru yüklendi.")

# =============================================================================
# 5) EMBEDDING & INDEX İŞLEMLERİ
# =============================================================================
def embed_single_text(text, model=EMBEDDING_MODEL):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if debug_mode:
                st.write(f"Embedding işlemi (deneme {attempt+1}): {text[:50]}...")
            response = openai.Embedding.create(model=model, input=[text])
            return response["data"][0]["embedding"]
        except openai.error.RateLimitError as e:
            if debug_mode:
                st.warning("RateLimitError oluştu, 5 saniye sonra tekrar deneniyor...")
                st.exception(e)
            time.sleep(5)
        except Exception as e:
            if debug_mode:
                st.error("embed_single_text fonksiyonunda hata:")
                st.exception(e)
            time.sleep(5)
    raise RuntimeError("embed_single_text için maksimum deneme sayısı aşıldı.")

def index_data(df):
    try:
        stats = index.describe_index_stats()
        vector_count = stats.get("total_vector_count", 0)
        if debug_mode:
            st.write("Indexteki mevcut vektör sayısı:", vector_count)
    except Exception as e:
        st.error("Index stats alınırken hata oluştu!")
        if debug_mode:
            st.exception(e)
        vector_count = 0

    if vector_count > 0:
        st.sidebar.info("Index zaten doldurulmuş. (Zaten indekslenmiş)")
        return

    st.info("Veriler Pinecone indeksine ekleniyor...")
    vectors = []
    for idx, row in df.iterrows():
        text = row["combined_text"]
        try:
            embedding = embed_single_text(text)
        except Exception as e:
            st.error(f"Embedding oluşturulurken hata (ID: {idx})")
            if debug_mode:
                st.exception(e)
            continue
        vector = {
            "id": str(idx),
            "values": embedding,
            "metadata": row.to_dict()
        }
        vectors.append(vector)
        if len(vectors) == BATCH_SIZE:
            try:
                index.upsert(vectors=vectors)
            except Exception as e:
                st.error("Batch upsert sırasında hata oluştu!")
                if debug_mode:
                    st.exception(e)
            vectors = []
    if vectors:
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            st.error("Final upsert sırasında hata oluştu!")
            if debug_mode:
                st.exception(e)
    st.success("Veriler başarıyla indekslendi.")

if not df.empty:
    with st.spinner("Veriler indeksleniyor..."):
        index_data(df)

# =============================================================================
# 6) BENZER SORU ÖNERİSİ FONKSİYONLARI
# =============================================================================
def query_similar_questions(query_text, top_k=5):
    try:
        query_embedding = embed_single_text(query_text)
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        if debug_mode:
            st.write("Pinecone'dan gelen yanıt:", query_response)
        return query_response.get("matches", [])
    except Exception as e:
        st.error("query_similar_questions sırasında hata oluştu:")
        if debug_mode:
            st.exception(e)
        return []

def get_recommendations(query_text, top_k=5):
    try:
        candidates = query_similar_questions(query_text, top_k=top_k)
        if not candidates:
            return []
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        recommendations = [{
            "soru_metni": cand["metadata"].get("soru_metni", "N/A"),
            "score": cand.get("score", 0.0)
        } for cand in sorted_candidates]
        return recommendations
    except Exception as e:
        if debug_mode:
            st.exception(e)
        return []

# =============================================================================
# 7) ANA KULLANICI ARAYÜZÜ (Soru Seçimi, Cevaplama & Öneriler)
# =============================================================================

# Oturum durumunda (session state) seçilen sorunun saklanması
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

st.sidebar.header("Soru Seçimi & Cevaplama")
st.sidebar.markdown(
    """
    **Adımlar:**
    1. Aşağıdaki arama kutusunu kullanarak soru bulun.
    2. Bir soru seçin.
    3. Cevap seçeneklerinden doğru cevabı işaretleyin.
    4. "Cevabı Gönder" butonuna tıklayın.
    """
)

if df.empty:
    st.error("Soru verisi mevcut değil.")
else:
    # Soru arama & filtreleme
    search_query = st.sidebar.text_input("Sorular arasında ara:")
    if search_query:
        filtered_df = df[df["soru_metni"].str.contains(search_query, case=False, na=False)]
    else:
        filtered_df = df

    if filtered_df.empty:
        st.sidebar.warning("Aramanıza uygun soru bulunamadı.")
        question_options = []
    else:
        question_options = filtered_df["soru_metni"].tolist()
    
    # Varsayılan seçimi session_state üzerinden ayarlıyoruz:
    if st.session_state.selected_question in question_options:
        default_index = question_options.index(st.session_state.selected_question)
    else:
        default_index = 0

    selected_question = st.sidebar.selectbox("Bir soru seçiniz:", question_options, index=default_index)
    st.session_state.selected_question = selected_question  # Güncelle

    # Ana içerik alanı: Soru & Cevap ve Ek Bilgiler olmak üzere iki sekme
    tab1, tab2 = st.tabs(["Soru & Cevap", "Ek Bilgiler"])
    
    with tab1:
        st.header("Soru")
        st.markdown(f"**{selected_question}**")
        try:
            selected_row = df[df["soru_metni"] == selected_question].iloc[0]
        except Exception as e:
            st.error("Seçilen soruya ait veriler alınırken hata oluştu!")
            if debug_mode:
                st.exception(e)
            st.stop()
    
        # Cevap seçeneklerini ayrıştırma
        try:
            choices = ast.literal_eval(selected_row["şıklar"])
        except Exception as e:
            if debug_mode:
                st.warning("Şıklar alanının literal değerlendirmesi başarısız oldu, alternatif yöntem kullanılıyor.")
                st.exception(e)
            choices = selected_row["şıklar"].split(",")
    
        st.subheader("Cevap Seçenekleri")
        user_choice = st.radio("Lütfen doğru cevabı seçiniz:", choices, key="answer_radio")
    
        if st.button("Cevabı Gönder", key="submit_answer"):
            correct_answer = selected_row["doğru_şık"].strip()
            if user_choice.strip() == correct_answer:
                st.success("Tebrikler, doğru cevap!")
                st.balloons()
            else:
                st.error("Maalesef, yanlış cevap.")
                st.info("Konuyu pekiştirmek için öneriler hazırlanıyor...")
                with st.spinner("Öneriler hazırlanıyor..."):
                    rec_list = get_recommendations(selected_question, top_k=5)
                if rec_list:
                    with st.expander("Önerilen Sorulara Göz Atın"):
                        st.markdown("**Önerilen Sorular:**")
                        for i, rec in enumerate(rec_list, start=1):
                            # Her öneriyi tıklanabilir buton olarak sunuyoruz.
                            if st.button(f"{i}. {rec['soru_metni']} (Benzerlik: {rec['score']:.3f})", key=f"rec_{i}"):
                                st.session_state.selected_question = rec["soru_metni"]
                                st.experimental_rerun()
                else:
                    st.write("Öneri üretilemedi, lütfen daha sonra tekrar deneyiniz.")
    
    with tab2:
        st.header("Seçilen Soru Hakkında Ek Bilgiler")
        st.markdown("Aşağıdaki bilgiler, bu sorunun hangi konu ve alt konuları kapsadığını, MEB müfredatı bilgilerini ve diğer detayları içerir:")
        extra_info = {
            "Çözüm": selected_row.get("çözüm", "Veri yok"),
            "Alt Konular": selected_row.get("alt_konular", "Veri yok"),
            "Konu": selected_row.get("konu", "Veri yok"),
            "MEB Kazanım": selected_row.get("meb_kazanım", "Veri yok"),
            "Taxonomy": selected_row.get("taxonomy", "Veri yok"),
            "Soru Türü": selected_row.get("soru_türü", "Veri yok"),
            "Zorluk": selected_row.get("difficulty", "Veri yok"),
            "Sınıf": selected_row.get("sinif", "Veri yok")
        }
        for key, value in extra_info.items():
            st.markdown(f"**{key}:** {value}")

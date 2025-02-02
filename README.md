MotiKoç canlı linki : https://motikoc-emin-durmus-eren-mollaoglu.streamlit.app/


# Soru Çözüm Öneri Sistemi

Bu proje, öğrencilere cevaplayamadıkları sorulara benzer sorular önererek konuları daha iyi kavramalarına yardımcı olan interaktif bir sistemdir. Hem soru çözüm önerileri sunan hem de öğrencilerin öğrenme süreçlerini destekleyen çeşitli özellikler içermektedir. Projede, Streamlit arayüzü, OpenAI API, Pinecone, NetworkX ve Matplotlib gibi modern teknolojiler kullanılmıştır.

---

## Özellikler

- **Soru Arama ve Seçimi:**  
  CSV dosyası içerisindeki sorular arasında arama yaparak istediğiniz soruyu seçebilirsiniz.

- **Cevaplama ve Geri Bildirim:**  
  Seçilen soruya ait cevap seçeneklerinden doğru cevabı işaretlediğinizde, sistem doğru/yanlış geri bildirimi verir ve doğru cevap durumunda kutlamalar (balonlar) gösterir.

- **Benzer Soru Önerileri:**  
  Yanlış cevap verdiğinizde, OpenAI tarafından oluşturulan embedding’ler ile Pinecone indeksinde arama yapılarak, benzer sorular önerilir.

- **Veri İndeksleme:**  
  Sorular, çeşitli metin alanları (soru metni, çözüm, alt konular, konu, MEB kazanım vb.) birleştirilerek OpenAI’nın embedding servisi ile vektörleştirilir ve Pinecone indeksine kaydedilir.

- **LLM ile Soru Analizi:**  
  Soru bilgileri, MEB müfredat verileri ve ek açıklamalar ile zenginleştirilmek üzere GPT-4 (veya GPT-3.5-turbo) modeli ile analiz edilir. Çıktı, belirli bir JSON formatında alınır.

- **Knowledge Graph Görselleştirmesi:**  
  Öğrenci profilleri, dersler, konular, videolar, sorular ve hafıza teknikleri arasındaki ilişkiler, NetworkX ve Matplotlib kullanılarak grafiksel olarak sunulur. Farklı senaryolar (ör. Emin ve Ahmet profilleri, benzer profil öneri mekanizması) için ayrı grafikler hazırlanmıştır.

---

## Teknolojiler

- [Streamlit](https://streamlit.io/)  
- [Pandas](https://pandas.pydata.org/)  
- [OpenAI API](https://openai.com/)  
- [Pinecone](https://pinecone.io/)  
- [tqdm](https://github.com/tqdm/tqdm)  
- [NetworkX](https://networkx.org/)  
- [Matplotlib](https://matplotlib.org/)

---

## Kurulum ve Çalıştırma

### Gereksinimler

- **Python 3.8** veya daha üstü sürüm  
- Gerekli Python paketlerini yüklemek için aşağıdaki komutu çalıştırın:

```bash
pip install streamlit pandas openai pinecone-client tqdm networkx matplotlib

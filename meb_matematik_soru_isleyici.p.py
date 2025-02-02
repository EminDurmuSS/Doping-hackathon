import os
import re
import json
import time
import logging
import pandas as pd
import openai

# API anahtarınızı ortam değişkeni üzerinden alın.
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY ortam değişkeni ayarlanmamış!")

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MEB müfredatlarının tanımlanması (9, 10, 11, 12. sınıflar için örnek metinler)
MEB_CURRICULUMS = {
    "9": """
MEB 9. Sınıf Matematik Öğretim Programı

1. Sayılar ve Cebir:
   - Sayı Kümeleri: Gerçek sayılar, tam sayılar, kesirli sayılar.
   - Üslü İfadeler: Üslü sayıların özellikleri, hesaplama yöntemleri.
   - Köklü İfadeler: Köklü ifadelerin özellikleri, sadeleştirme işlemleri.
   - Çarpanlara Ayırma: Polinomların çarpanlara ayrılması, sadeleştirme.
   - Denklemler ve Eşitsizlikler: Birinci dereceden denklemler ve eşitsizlikler.

2. Geometri ve Ölçme:
   - Temel Geometrik Kavramlar: Nokta, doğru, düzlem.
   - Doğrular ve Açılar: Paralel/dik doğrular, açı ölçümleri.
   - Üçgenler: Üçgen çeşitleri, açı-kenar ilişkileri, alan hesaplamaları.
   - Dörtgenler ve Çember: Dörtgenlerin özellikleri, çember ve yay kavramları.

3. Veri, Sayma ve Olasılık:
   - Veri Analizi: Veri toplama, düzenleme, yorumlama.
   - Olasılık: Temel olasılık kavramları ve basit olaylar.
""",
    "10": """
MEB 10. Sınıf Matematik Öğretim Programı

1. Sayılar ve Cebir:
   - Rasyonel Sayılar: Kesir ve ondalık sayılar.
   - Polinomlar: Polinom kavramı, dereceler, temel işlemler.
   - Üslü ve Köklü İfadeler: İleri üslü ve köklü ifadeler.
   - Denklemler ve Eşitsizlikler: İkinci dereceden denklemler, mutlak değer denklemleri.

2. Fonksiyonlar:
   - Fonksiyon Kavramı: Tanım, grafik çizimi, fonksiyon türleri.
   - Lineer, Kuadratik ve Rasyonel Fonksiyonlar: Özellikler ve analizler.

3. Geometri:
   - Analitik Geometri: Nokta, doğru, düzlem denklemleri.
   - Üçgen, Dörtgen ve Çember: Geometrik özellikler ve hesaplamalar.

4. İstatistik ve Olasılık:
   - Veri Analizi: İstatistiksel yöntemler, grafikler.
   - Olasılık: Karmaşık olayların olasılık hesaplamaları.
""",
    "11": """
MEB 11. Sınıf Matematik Öğretim Programı

1. Fonksiyonlar ve Cebir:
   - Fonksiyonlar: Tanım, çeşitleri, birleşim ve ters fonksiyon.
   - Polinomlar: Polinom bölme, katsayılar ve kök analizi.
   - Rasyonel Fonksiyonlar: Asimptotlar, tanım kümesi.

2. Trigonometri:
   - Temel Trigonometri: Açılar ve trigonometrik oranlar.
   - İleri Trigonometri: Denklemler, kimlikler ve dönüşümler.

3. Analitik Geometri:
   - Doğrular ve Düzlemler: Analitik yöntemler, kesişim noktaları.
   - Konikler: Parabol, elips, hiperbol ve daire.
   - Vektörler: Vektör işlemleri, doğrultu ve norm kavramları.

4. İstatistik ve Olasılık:
   - İstatistik: Dağılımlar, ortalama, varyans.
   - Olasılık: Rastgele olaylar, kombinatorik analiz.
""",
    "12": """
MEB 12. Sınıf Matematik Öğretim Programı

1. İnceleme ve Analiz:
   - Limit ve Süreklilik: Limit kavramı, süreklilik ve limit hesaplamaları.
   - Türev: Türev tanımı, kurallar ve uygulamalar.
   - İntegral: Belirli ve belirsiz integraller, alan hesaplamaları.
   - Diferansiyel Denklemler: Temel kavramlar ve çözüm yöntemleri.

2. İleri Fonksiyonlar ve Cebir:
   - Matris ve Determinant: Matris işlemleri, determinant hesaplamaları.
   - İleri Polinomlar: Polinom fonksiyonlar ve kök analizi.
   - Logaritmik ve Üstel Fonksiyonlar: İleri düzey hesaplamalar.

3. Geometri ve Analitik Geometri:
   - Uzay Geometrisi: Uzayda nokta, doğru, düzlem ilişkileri.
   - Dönüşümler ve Vektörler: Koordinat dönüşümleri, vektör uzayları.

4. Olasılık ve İstatistik:
   - İleri Olasılık: Olaylar, koşullu olasılık, dağılımlar.
   - İstatistiksel Yöntemler: Veri analizi, regresyon, hipotez testleri.
"""
}

def get_meb_curriculum(sinif_value: str) -> str:
    """
    Verilen sınıf bilgisinden (örn. '9. sınıf', '10', '11', '12') ilgili müfredat bilgisini döndürür.
    Varsayılan olarak 9. sınıf müfredatı döner.
    """
    grade_match = re.search(r'(\d+)', sinif_value)
    grade = grade_match.group(1) if grade_match else "9"
    curriculum = MEB_CURRICULUMS.get(grade, MEB_CURRICULUMS["9"])
    return curriculum.strip()

def build_prompt(row: dict) -> str:
    """
    CSV'deki soru satırındaki bilgileri ve ilgili MEB müfredatını kullanarak LLM'e gönderilecek promptu oluşturur.
    Çıktıda istenen tüm alanlar yer almalıdır.
    """
    sinif_value = row.get("sinif", "9")
    curriculum_text = get_meb_curriculum(str(sinif_value))
    
    # Soru türlerinin açıklamaları eklendi:
    soru_turleri_aciklama = """
**Soru Türleri ve Açıklamaları**:
1. Yorumlama Soruları: Verilen bilgi veya metni analiz ederek yorum yapmanızı gerektiren sorulardır. Grafik, tablo veya paragraf yorumlama bu kategoriye girer.
2. Problem Çözme Soruları: Matematiksel veya mantıksal problemleri çözmenizi isteyen sorulardır. Genellikle birden fazla adımda çözüm gerektirir.
3. Uygulama Soruları: Teorik bilgilerinizi pratik durumlara uygulamanızı isteyen sorulardır. Örneğin, fiziksel bir prensibi gerçek bir olaya uygulamak gibi.
4. Analiz ve Sentez Soruları: Birden fazla bilgiyi bir araya getirerek analiz yapmanızı ve yeni bir sonuç çıkarmanızı gerektiren sorulardır.
5. Eleştirel Düşünme Soruları: Verilen argümanları değerlendirmenizi ve eleştirmenizi isteyen sorulardır. Doğruyu yanlıştan ayırt etme becerisi önemlidir.

Lütfen 'soru_türü' alanını yukarıdaki kategorilerden en uygun olanı seçerek doldurun.
"""

    prompt = f"""
Aşağıdaki matematik sorusunu ilgili MEB müfredatına uygun olarak analiz et.
Eksik bilgileri tamamlayarak ve gerekirse yorum ekleyerek aşağıdaki formatta **sadece JSON** çıktısı üret.
Lütfen çıktı, yalnızca aşağıdaki anahtarları içermelidir:

{{
  "soru_id": "<soru id'si>",
  "soru_metni": "<soru metni>",
  "şıklar": <şıklar listesi>,
  "doğru_şık": "<doğru şık>",
  "çözüm": "<çözüm açıklaması>",
  "alt_konular": <alt konular listesi>,
  "meb_kazanım": "<MEB kazanım>",
  "taxonomy": "<taxonomy>",
  "soru_türü": "<soru türü>",
  "konu": "<konu>",
  "cozum_suresi": "<çözüm süresi>",
  "difficulty": "<zorluk seviyesi>",
  "sinif": "<sınıf>",
  "sik_yapilan_hatalar": <sık yapılan hatalar listesi>,
  "matematik_formulu": "<matematik formülü>",
  "ek_not": "Varsa ek açıklamalar"
}}

{soru_turleri_aciklama}

Soru bilgileri:
soru_id: {row.get('soru_id', '').strip()}
soru_metni: {row.get('soru_metni', '').strip()}
şıklar: {row.get('şıklar', '').strip()}
doğru_şık: {row.get('doğru_şık', '').strip()}
çözüm: {row.get('çözüm', '').strip()}
alt_konular: {row.get('alt_konular', '').strip()}
meb_kazanım: {row.get('meb_kazanım', '').strip()}
taxonomy: {row.get('taxonomy', '').strip()}
soru_türü: {row.get('soru_türü', '').strip()}
konu: {row.get('konu', '').strip()}
cozum_suresi: {row.get('cozum_suresi', '').strip()}
difficulty: {row.get('difficulty', '').strip()}
sinif: {row.get('sinif', '').strip()}
sik_yapilan_hatalar: {row.get('sik_yapilan_hatalar', '').strip()}
matematik_formulu: {row.get('matematik_formulu', '').strip()}

Ayrıca, ilgili MEB müfredatı (Sınıf: {sinif_value}) bilgisi aşağıdadır:
---------------------------------------------------------
{curriculum_text}
---------------------------------------------------------

Lütfen **sadece JSON formatında** çıktı üret.
"""
    return prompt

def process_question(row: dict, max_retries: int = 3, delay: int = 2) -> dict:
    """
    Verilen soru satırını LLM'e gönderir, analiz ettirir ve JSON çıktısı olarak döndürür.
    Hata durumunda belirlenen sayıda yeniden deneme yapar.
    """
    prompt = build_prompt(row)
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Veya ihtiyaca göre "gpt-3.5-turbo"
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            logging.info(f"Soru {row.get('soru_id', 'Bilinmiyor')} başarıyla işlendi.")
            return result
        except Exception as e:
            logging.error(f"Deneme {attempt}/{max_retries} - Soru {row.get('soru_id', 'Bilinmiyor')} işlenirken hata: {e}")
            time.sleep(delay)
    logging.warning(f"Soru {row.get('soru_id', 'Bilinmiyor')} işlenemedi.")
    return {}

def main():
    # CSV dosya yolunu ihtiyacınıza göre güncelleyin.
    csv_file = "path/to/your/sorular.csv"  # Örn: "./data/sorular.csv"
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logging.error(f"CSV dosyası okunurken hata oluştu: {e}")
        return

    extracted_data = []
    # Her soru için LLM çağrısı yapılır.
    for index, row in df.iterrows():
        soru_id = row.get("soru_id", index)
        logging.info(f"İşlenen Soru ID: {soru_id}")
        processed = process_question(row)
        if processed:
            # İsteğe bağlı olarak soru_id bilgisini ekleyin.
            processed["soru_id"] = soru_id
            extracted_data.append(processed)
        else:
            logging.warning(f"Soru {soru_id} atlandı.")

    if extracted_data:
        result_df = pd.DataFrame(extracted_data)
        output_csv = "extracted_questions.csv"
        try:
            result_df.to_csv(output_csv, index=False)
            logging.info(f"Çıktılar '{output_csv}' dosyasına başarıyla kaydedildi.")
        except Exception as e:
            logging.error(f"Sonuçların CSV dosyasına yazılırken hata: {e}")
    else:
        logging.warning("Hiçbir soru işlenemedi, çıktı oluşturulmadı.")

if __name__ == "__main__":
    main()

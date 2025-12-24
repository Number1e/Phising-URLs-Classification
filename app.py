import pickle
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import math
import re
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_tabular import TabularModel
import torch.nn.functional as F
import pytorch_lightning


try:
    from pytorch_lightning.utilities import model_helpers
    if not hasattr(model_helpers, '_ModuleMode'):
        # Kita buat kelas palsu untuk menipu Pickle
        class _ModuleMode:
            pass
        # Kita suntikkan ke dalam library yang sedang berjalan
        setattr(model_helpers, '_ModuleMode', _ModuleMode)
        print("✅ Patch '_ModuleMode' berhasil diterapkan.")
except ImportError:
    pass # Jika versi library terlalu beda, abaikan
except Exception as e:
    print(f"⚠️ Gagal menerapkan patch: {e}")

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Phishing Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Sistem Deteksi Phishing URL")
st.markdown("""
Aplikasi ini menggunakan **5 Model Deep Learning** berbeda untuk mendeteksi apakah sebuah URL berbahaya (Phishing) atau aman.
""")

# ==========================================
# 2. HELPER: EKSTRAKSI FITUR (PENTING!)
# ==========================================
# Kita harus mengubah URL mentah menjadi fitur angka agar bisa dibaca oleh
# Model Neural Network, TabNet, dan FT-Transformer.

def calculate_entropy(text):
    if not text:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(text.count(chr(x))) / len(text)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def extract_features(url):
    # Ini harus SAMA PERSIS dengan preprocessing saat training
    features = {}
    
    # 1. URL Length
    features['url_length'] = len(url)
    
    # 2. Num Dots
    features['num_dots'] = url.count('.')
    
    # 3. Has HTTPS
    features['has_https'] = 1 if "https" in url else 0
    
    # 4. Has IP
    ip_pattern = r'(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])'
    features['has_ip'] = 1 if re.search(ip_pattern, url) else 0
    
    # 5. Num Subdirs
    features['num_subdirs'] = url.count('/')
    
    # 6. Num Params
    features['num_params'] = url.count('?') + url.count('&')
    
    # 7. Suspicious Words
    suspicious = ["login", "signin", "bank", "account", "update", "verify", "secure", "ebay", "paypal"]
    features['suspicious_words'] = sum(1 for word in suspicious if word in url.lower())
    
    # 8. Special Char Count
    features['special_char_count'] = sum(1 for c in url if not c.isalnum())
    
    # 9. Digits Count
    features['digits_count'] = sum(c.isdigit() for c in url)
    
    # 10. Entropy
    features['entropy'] = calculate_entropy(url)
    
    # 11. TLD (Categorical) - Sederhana: ambil bagian terakhir setelah titik
    try:
        parsed = urlparse(url)
        if parsed.netloc:
            domain_parts = parsed.netloc.split('.')
            if len(domain_parts) > 1:
                features['tld'] = "" + domain_parts[-1]
            else:
                features['tld'] = "unknown"
        else:
            # Fallback jika url tidak punya scheme (http/https)
            domain_parts = url.split('/')[0].split('.')
            if len(domain_parts) > 1:
                features['tld'] = "." + domain_parts[-1]
            else:
                features['tld'] = "unknown"
    except:
        features['tld'] = "unknown"

    return features

# ==========================================
# 3. LOAD MODEL (CACHED)
# ==========================================
import os
BASE_PATH = os.path.join(os.path.dirname(__file__), 'models')

@st.cache_resource
def load_nn_model():
    # Load Model Keras
    model_path = os.path.join(BASE_PATH, "neural_network_base.h5")
    
    # PERHATIKAN NAMA FILE: Pastikan ejaan 'scaler_nn.pkl' atau 'scale_nn.pkl' 
    # sesuai dengan nama file fisik di folder models Anda.
    scaler_path = os.path.join(BASE_PATH, "scaler_nn.pkl") 
    
    model = None
    scaler = None

    # 1. Load Model Keras
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat Model NN: {e}")
        return None, None

    # 2. Load Scaler (Gunakan JOBLIB, bukan Pickle)
    try:
        # Joblib lebih robust untuk objek scikit-learn
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat Scaler: {e}")
        st.warning("Coba cek: 1. Apakah file korup? 2. Apakah nama file benar?")
        return None, None

    return model, scaler
@st.cache_resource
def load_transformer_model(model_name_or_path):
    # 1. Tentukan Konfigurasi Berdasarkan Pilihan
    if "distilbert" in model_name_or_path.lower():
        folder_name = "distilbert_phishing"
        base_model_id = "distilbert-base-uncased" # ID Asli di HuggingFace
    elif "canine" in model_name_or_path.lower():
        folder_name = "canine_phishing"
        base_model_id = "google/canine-s"
    else:
        st.error("Model tidak dikenali.")
        return None, None

    # Path Folder Lokal
    save_dir = os.path.join(BASE_PATH, folder_name)

    # 2. LOAD TOKENIZER (DARI INTERNET / BASE MODEL)
    # Karena folder Anda tidak punya vocab.txt, kita ambil dari base modelnya.
    try:
        print(f"🔄 Mengunduh Tokenizer standar untuk {base_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    except Exception as e:
        st.error(f"❌ Gagal memuat Tokenizer (Butuh Internet): {e}")
        return None, None

    # 3. LOAD MODEL (DARI LOKAL - SAFETENSORS)
    # Kita tambahkan parameter use_safetensors=True karena file Anda .safetensors
    try:
        print(f"📂 Memuat Model Lokal dari: {save_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(
            save_dir, 
            local_files_only=True,
            use_safetensors=True # <--- PENTING: Karena file Anda .safetensors bukan .bin
        )
    except Exception as e:
        st.error(f"❌ Gagal memuat Model Weights: {e}")
        st.warning(f"Pastikan file 'model.safetensors' dan 'config.json' ada di {save_dir}")
        return None, None

    return tokenizer, model
@st.cache_resource
def load_tabular_model(model_type):
    from sklearn.preprocessing import LabelEncoder
    
    try:
        # 1. Tentukan Folder
        if model_type == "TabNet":
            folder_name = "tabnet_model"
        else:
            folder_name = "ft_transformer_model"
            
        path = os.path.join(BASE_PATH, folder_name)
        
        # 2. Cek Folder
        if not os.path.exists(path):
             st.error(f"❌ Folder Model tidak ditemukan di: {path}")
             return None

        # 3. Load Model
        model = TabularModel.load_model(path)
        
        # ====================================================
        # 🔧 HARD FIX: PAKSA LABEL ENCODER JADI [0, 1]
        # ====================================================
        # Kita tidak peduli apa isi file aslinya (karena sering korup).
        # Kita tahu pasti ini Binary Classification (0 dan 1).
        # Jadi kita buat Encoder baru secara manual.
        
        try:
            print(f"🔧 Menerapkan Hard-Fix LabelEncoder untuk {model_type}...")
            
            # Buat Encoder Baru
            le = LabelEncoder()
            
            # PAKSA dia belajar angka 0 dan 1
            # Ini memastikan model bisa menerjemahkan prediksi index 0 dan 1
            le.classes_ = np.array([0, 1]) 
            le.fit([0, 1]) 
            
            # Suntikkan paksa ke dalam model
            # Kita timpa apapun yang ada sebelumnya (list rusak, objek kosong, dll)
            if hasattr(model, 'datamodule'):
                model.datamodule.label_encoder = le
                print("✅ LabelEncoder berhasil di-hardcode ke [0, 1].")
            else:
                print("⚠️ Model tidak memiliki atribut datamodule (Aneh, tapi kita lanjut).")

        except Exception as patch_error:
            print(f"⚠️ Gagal menerapkan patch: {patch_error}")

        return model

    except Exception as e:
        st.error(f"❌ Error Tabular Model ({model_type}): {e}")
        return None
# ==========================================
# 4. USER INTERFACE & LOGIC
# ==========================================

# Sidebar
st.sidebar.header("⚙️ Konfigurasi Model")
model_choice = st.sidebar.selectbox(
    "Pilih Model Deteksi:",
    [
        "Neural Network (Base)",
        "DistilBERT (Text-Based)",
        "CANINE (Character-Based)",
        "TabNet (Tabular DL)",
        "FT-Transformer (Tabular DL)"
    ]
)

EXPECTED_COLS = [
    'url_length', 'num_dots', 'has_https', 'has_ip', 
    'num_subdirs', 'num_params', 'suspicious_words', 'tld', 
    'special_char_count', 'digits_count', 'entropy'
]

# ==========================================
# 5. MAIN AREA (UI & LOGIC) - VERSI FIXED
# ==========================================

# ==========================================
# 5. MAIN AREA (FIXED: RE-ATTACH TLD)
# ==========================================

url_input = st.text_input("🔗 Masukkan URL yang ingin dianalisis:", placeholder="http://tokopedia.com")

if st.button("🔍 Analisis URL"):
    if not url_input:
        st.warning("Mohon masukkan URL terlebih dahulu.")
    else:
        st.info(f"Sedang menganalisis menggunakan model: **{model_choice}**...")
        
        # --- LOGIKA PREDIKSI ---
        prediction_score = 0
        
        # GROUP A: TABULAR MODELS
        if model_choice in ["Neural Network (Base)", "TabNet (Tabular DL)", "FT-Transformer (Tabular DL)"]:
            
            # A. Definisi Urutan Kolom
            EXPECTED_COLS = [
                'url_length', 'num_dots', 'has_https', 'has_ip', 
                'num_subdirs', 'num_params', 'suspicious_words', 'tld', 
                'special_char_count', 'digits_count', 'entropy'
            ]

            # B. Ekstrak Fitur
            feats = extract_features(url_input) 
            df_single = pd.DataFrame([feats])

            # C. Samakan Urutan Kolom
            missing_cols = [c for c in EXPECTED_COLS if c not in df_single.columns]
            if missing_cols:
                st.error(f"❌ Error Kolom Hilang: {missing_cols}")
                st.stop()
            
            df_single = df_single[EXPECTED_COLS]

            # D. SCALING DATA
            # Load Scaler dari NN
            load_result = load_nn_model()
            
            if load_result is None or load_result[1] is None:
                st.error("❌ Gagal memuat Scaler (scale_nn.pkl).")
                st.stop()
            
            _, scaler = load_result
            
            # Proses Scaling
            # Ambil kolom numerik saja (drop TLD string)
            X_raw_numeric = df_single.drop(columns=['tld'])
            
            try:
                X_scaled_array = scaler.transform(X_raw_numeric)
            except Exception as e:
                st.error(f"❌ Gagal melakukan Scaling: {e}")
                st.stop()
            
            # Buat DataFrame Scaled
            df_scaled = pd.DataFrame(X_scaled_array, columns=X_raw_numeric.columns)

            # --- PERBAIKAN DI SINI: MASUKKAN KEMBALI KOLOM TLD ---
            # TabNet & FT-Transformer butuh kolom 'tld' meskipun itu string
            # Neural Network TIDAK butuh 'tld'
            df_scaled['tld'] = df_single['tld'].values 
            # -----------------------------------------------------

            # E. PREDIKSI
            if model_choice == "Neural Network (Base)":
                model, _ = load_nn_model()
                if model:
                    # NN murni angka, jadi kita pakai array numerik saja (tanpa TLD)
                    prob = model.predict(X_scaled_array)[0][0]
                    prediction_score = float(prob)

            elif model_choice == "TabNet (Tabular DL)":
                model = load_tabular_model("TabNet")
                if model:
                    # TabNet pakai df_scaled (yang sudah ada TLD-nya lagi)
                    pred = model.predict(df_scaled)
                    if '1_probability' in pred.columns:
                        prediction_score = pred['1_probability'].values[0]
                    elif 'prediction' in pred.columns:
                        prediction_score = float(pred['prediction'].values[0])

            elif model_choice == "FT-Transformer (Tabular DL)":
                model = load_tabular_model("FT-Transformer")
                if model:
                    # FT-Transformer pakai df_scaled (yang sudah ada TLD-nya lagi)
                    pred = model.predict(df_scaled)
                    if '1_probability' in pred.columns:
                        prediction_score = pred['1_probability'].values[0]
                    elif 'prediction' in pred.columns:
                        prediction_score = float(pred['prediction'].values[0])

        # GROUP B: TEXT MODELS
        else:
            if model_choice == "DistilBERT (Text-Based)":
                tokenizer, model = load_transformer_model("distilbert")
            else:
                tokenizer, model = load_transformer_model("canine")
            
            if tokenizer and model:
                inputs = tokenizer(url_input, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                prediction_score = probs[0][1].item()

        # --- TAMPILKAN HASIL ---
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Skor Probabilitas Phishing", f"{prediction_score:.2%}")
            
        with col2:
            if prediction_score > 0.5:
                st.error("🚨 KESIMPULAN: PHISHING DETECTED")
                st.write("URL ini memiliki indikasi kuat berbahaya.")
            else:
                st.success("✅ KESIMPULAN: AMAN (BENIGN)")
                st.write("URL ini terlihat aman.")
        
        st.progress(float(prediction_score))
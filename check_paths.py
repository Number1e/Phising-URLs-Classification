import os

# Ganti ini jika nama folder model Anda berbeda
BASE_PATH = os.path.join(os.path.dirname(__file__), 'models')

print(f"📂 Memeriksa folder: {BASE_PATH}\n")

# 1. Cek Neural Network
nn_path = os.path.join(BASE_PATH, "neural_network_base.h5")
if os.path.exists(nn_path):
    print("✅ Neural Network ditemukan.")
else:
    print(f"❌ Neural Network GAGAL. File tidak ditemukan di: {nn_path}")

# 2. Cek TabNet
tabnet_path = os.path.join(BASE_PATH, "tabnet_model", "config.yml")
if os.path.exists(tabnet_path):
    print("✅ TabNet Config ditemukan.")
else:
    print(f"❌ TabNet GAGAL. config.yml tidak ditemukan di: {tabnet_path}")
    print("   👉 Cek apakah folder ter-extract ganda (tabnet_model/tabnet_model/...)")

# 3. Cek FT-Transformer
ft_path = os.path.join(BASE_PATH, "ft_transformer_model", "config.yml")
if os.path.exists(ft_path):
    print("✅ FT-Transformer Config ditemukan.")
else:
    print(f"❌ FT-Transformer GAGAL. config.yml tidak ditemukan di: {ft_path}")

# 4. Cek DistilBERT
bert_path = os.path.join(BASE_PATH, "distilbert_phishing", "config.json")
if os.path.exists(bert_path):
    print("✅ DistilBERT Config ditemukan.")
else:
    print(f"❌ DistilBERT GAGAL. config.json tidak ditemukan di: {bert_path}")
    print("   👉 Pastikan nama folder 'distilbert_phishing' sesuai dengan yang Anda download.")

# 5. Cek CANINE
canine_path = os.path.join(BASE_PATH, "canine_phishing", "config.json")
if os.path.exists(canine_path):
    print("✅ CANINE Config ditemukan.")
else:
    print(f"❌ CANINE GAGAL. config.json tidak ditemukan di: {canine_path}")
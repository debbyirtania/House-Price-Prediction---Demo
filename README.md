# End-to-End Prediksi Harga Rumah (XGBoost + FastAPI)

Project ini mencakup:
1. Pembuatan data dummy untuk train/test.
2. Training model XGBoost + hyperparameter tuning untuk mengurangi overfitting.
3. Deployment inference menggunakan FastAPI (lokal + Render).

## Struktur Folder

```bash
.
├── artifacts/
│   ├── data/
│   └── model/
├── src/
│   ├── app.py
│   ├── generate_data.py
│   └── train.py
├── render.yaml
├── requirements.txt
└── README.md
```

## Step-by-Step Eksekusi Lokal

### Step 1 - Buat virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### Step 2 - Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 - Generate dummy data train & test

```bash
python src/generate_data.py
```

Output:
- `artifacts/data/train.csv`
- `artifacts/data/test.csv`

### Step 4 - Training + tuning XGBoost

```bash
python src/train.py
```

Output:
- Menampilkan `best_params` hasil tuning.
- Menampilkan metrik train vs test (RMSE/MAE/R2) untuk cek overfitting.
- Menyimpan model ke `artifacts/model/xgboost_harga_rumah.joblib`
- Menyimpan metadata ke `artifacts/model/metadata.json`

### Step 5 - Jalankan FastAPI lokal

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### Step 6 - Akses lokal

- UI: `http://localhost:8000/`
- Swagger: `http://localhost:8000/docs`

## Deploy ke Render (Tanpa ngrok)

### Prasyarat

1. Project sudah di-push ke GitHub.
2. File model tersedia di repo:
- `artifacts/model/xgboost_harga_rumah.joblib`
- `artifacts/model/metadata.json`

### Opsi A - Deploy via `render.yaml` (direkomendasikan)

1. Buka Render -> New -> Blueprint.
2. Pilih repo GitHub project ini.
3. Render akan membaca `render.yaml` otomatis.
4. Tunggu deploy selesai, lalu buka URL publik Render.

Catatan kompatibilitas:
- Project ini dipin ke Python `3.11` melalui file `runtime.txt` agar dependency data science (pandas/xgboost) stabil saat build di Render.

### Opsi B - Deploy manual di dashboard Render

1. Render -> New -> Web Service.
2. Connect ke repo GitHub kamu.
3. Isi konfigurasi:
- Runtime: `Python`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
4. Deploy.

Setelah live:
- UI: `https://<nama-service>.onrender.com/`
- Swagger: `https://<nama-service>.onrender.com/docs`

Catatan:
- Di free plan Render, service bisa sleep saat idle.
- Berbeda dengan ngrok, URL Render tetap (lebih cocok untuk demo publik jangka panjang).

## Test Endpoint

Cek health:

```bash
curl http://127.0.0.1:8000/health
```

Prediksi harga rumah:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "luas_bangunan": 150,
  "luas_tanah": 210,
  "jumlah_kamar": 4,
  "jumlah_kamar_mandi": 3,
  "usia_rumah": 8,
  "jarak_ke_pusat_kota": 6.5
}'
```
### Open Via Website ngrok
https://bossily-expressible-whitley.ngrok-free.dev/

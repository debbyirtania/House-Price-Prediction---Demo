from __future__ import annotations

"""
# House Price Prediction API (FastAPI)

File ini adalah layer deployment untuk model XGBoost prediksi harga rumah.

## Alur singkat
1. Startup aplikasi: load model + metadata dari folder `artifacts/model`.
2. Endpoint `/health`: cek API hidup dan model berhasil dimuat.
3. Endpoint `/predict`: validasi input -> urutkan fitur -> prediksi -> return JSON.
"""

import json
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Lokasi artifact hasil training.
MODEL_PATH = Path("artifacts/model/xgboost_harga_rumah.joblib")
METADATA_PATH = Path("artifacts/model/metadata.json")

# Inisialisasi aplikasi FastAPI.
app = FastAPI(title="House Price Prediction API", version="1.0.0")

# Variabel global untuk menyimpan model yang sudah diload saat startup.
model = None
metadata = None


class HouseFeatures(BaseModel):
    """Schema input request untuk endpoint /predict."""

    # Validasi dasar:
    # - `gt=0`  : nilai harus > 0
    # - `ge=0/1`: nilai harus >= 0 atau >= 1
    luas_bangunan: float = Field(..., gt=0)
    luas_tanah: float = Field(..., gt=0)
    jumlah_kamar: int = Field(..., ge=1)
    jumlah_kamar_mandi: int = Field(..., ge=1)
    usia_rumah: int = Field(..., ge=0)
    jarak_ke_pusat_kota: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    """Schema output response untuk endpoint /predict."""

    prediksi_harga_rumah: float


def load_artifacts():
    """
    Load model dan metadata dari disk.

    Return:
    - (None, None) jika artifact belum tersedia.
    - (loaded_model, loaded_metadata) jika tersedia.
    """
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None

    # Model biner (.joblib) hasil training.
    loaded_model = joblib.load(MODEL_PATH)

    # Metadata JSON berisi daftar fitur + metrik training.
    loaded_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return loaded_model, loaded_metadata


@app.on_event("startup")
def startup_event():
    """Hook startup: dipanggil sekali saat server FastAPI dinyalakan."""
    global model, metadata
    model, metadata = load_artifacts()


@app.get("/health", include_in_schema=False)
def health_check():
    """Endpoint monitoring sederhana untuk health status API."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """Simple web UI: input fitur rumah lalu jalankan prediksi."""
    return """
<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Prediksi Harga Rumah</title>
  <style>
    :root {
      --bg: #f4f8f6;
      --card: #ffffff;
      --ink: #13302a;
      --muted: #4f6962;
      --line: #d8e4df;
      --accent: #0f8b6d;
      --accent-dark: #0a6b54;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: radial-gradient(circle at 10% 20%, #e3f2ec 0%, var(--bg) 45%, #f7fbf9 100%);
      color: var(--ink);
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .card {
      width: 100%;
      max-width: 720px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 16px 36px rgba(13, 49, 38, 0.08);
      padding: 24px;
    }
    h1 { margin: 0 0 8px; font-size: 1.6rem; }
    p { margin: 0 0 20px; color: var(--muted); }
    form {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .field { display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 0.9rem; font-weight: 600; color: var(--muted); }
    input {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      font-size: 0.95rem;
      outline: none;
    }
    input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 139, 109, 0.15);
    }
    .actions {
      grid-column: 1 / -1;
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 6px;
    }
    button {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 10px 16px;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { background: var(--accent-dark); }
    .result {
      margin-top: 18px;
      padding: 14px;
      border-radius: 10px;
      border: 1px dashed var(--line);
      background: #f7fcfa;
      font-size: 1rem;
      color: var(--ink);
      min-height: 52px;
      display: flex;
      align-items: center;
    }
    @media (max-width: 640px) {
      form { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <section class="card">
    <h1>Prediksi Harga Rumah</h1>
    <p>Isi fitur rumah di bawah lalu klik <b>Prediksi</b>.</p>
    <form id="predictForm">
      <div class="field">
        <label for="luas_bangunan">Luas Bangunan (m2)</label>
        <input id="luas_bangunan" name="luas_bangunan" type="number" min="1" step="0.01" value="150" required />
      </div>
      <div class="field">
        <label for="luas_tanah">Luas Tanah (m2)</label>
        <input id="luas_tanah" name="luas_tanah" type="number" min="1" step="0.01" value="210" required />
      </div>
      <div class="field">
        <label for="jumlah_kamar">Jumlah Kamar</label>
        <input id="jumlah_kamar" name="jumlah_kamar" type="number" min="1" step="1" value="4" required />
      </div>
      <div class="field">
        <label for="jumlah_kamar_mandi">Jumlah Kamar Mandi</label>
        <input id="jumlah_kamar_mandi" name="jumlah_kamar_mandi" type="number" min="1" step="1" value="3" required />
      </div>
      <div class="field">
        <label for="usia_rumah">Usia Rumah (tahun)</label>
        <input id="usia_rumah" name="usia_rumah" type="number" min="0" step="1" value="8" required />
      </div>
      <div class="field">
        <label for="jarak_ke_pusat_kota">Jarak ke Pusat Kota (km)</label>
        <input id="jarak_ke_pusat_kota" name="jarak_ke_pusat_kota" type="number" min="0" step="0.01" value="6.5" required />
      </div>
      <div class="actions">
        <button type="submit">Prediksi</button>
      </div>
    </form>
    <div class="result" id="result">Belum ada prediksi.</div>
  </section>

  <script>
    const form = document.getElementById("predictForm");
    const result = document.getElementById("result");

    function toIDR(number) {
      return new Intl.NumberFormat("id-ID", {
        style: "currency",
        currency: "IDR",
        maximumFractionDigits: 0
      }).format(number);
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      result.textContent = "Memproses prediksi...";

      const payload = {
        luas_bangunan: Number(form.luas_bangunan.value),
        luas_tanah: Number(form.luas_tanah.value),
        jumlah_kamar: Number(form.jumlah_kamar.value),
        jumlah_kamar_mandi: Number(form.jumlah_kamar_mandi.value),
        usia_rumah: Number(form.usia_rumah.value),
        jarak_ke_pusat_kota: Number(form.jarak_ke_pusat_kota.value),
      };

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) {
          result.textContent = data.detail || "Terjadi error saat prediksi.";
          return;
        }
        result.textContent = "Estimasi harga rumah: " + toIDR(data.prediksi_harga_rumah);
      } catch (_) {
        result.textContent = "Gagal terhubung ke server.";
      }
    });
  </script>
</body>
</html>
"""


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HouseFeatures):
    """
    Endpoint prediksi harga rumah.

    """
    try:
        if model is None or metadata is None:
            raise HTTPException(
                status_code=503,
                detail="Model belum tersedia. Jalankan: python src/generate_data.py lalu python src/train.py",
            )

        # Convert pydantic object -> dict python biasa.
        input_df = payload.model_dump()

        # Penting: urutan fitur harus sama dengan saat model dilatih.
        features = [input_df[col] for col in metadata["features"]]
        prediction = model.predict([features])[0]
        return {"prediksi_harga_rumah": round(float(prediction), 2)}
    except HTTPException:
        # Lempar ulang error FastAPI (contoh: 503) tanpa diubah.
        raise
    except Exception as exc:
        # Error tak terduga diubah jadi HTTP 500 agar konsisten di API response.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

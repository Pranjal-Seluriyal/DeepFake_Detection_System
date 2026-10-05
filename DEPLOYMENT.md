# DeepTrust AI &mdash; Production Deployment Guide

This guide walks you through deploying **DeepTrust AI** to a production cloud environment with a managed **PostgreSQL** database and **Cloudinary** object storage.

---

## Architecture Overview

```
                      +-----------------------------------+
                      |      User Browser / Mobile        |
                      |  (Camera API / Desktop / Tablet)  |
                      +-----------------+-----------------+
                                        | HTTPS
                                        v
                      +-----------------------------------+
                      |   Cloud Host (Render / Railway)   |
                      |   Gunicorn WSGI + Flask Backend   |
                      |   PyTorch Forensic Inference      |
                      +---------+---------------+---------+
                                |               |
           Persistent Data (SQL)|               | Media Uploads
                                v               v
                +--------------------+   +--------------------+
                |  Managed Postgres  |   |     Cloudinary     |
                |  (Supabase / Neon) |   | Cloud Storage CDN  |
                +--------------------+   +--------------------+
```

---

## 1. Prerequisites & Accounts Needed

1. **GitHub Account**: To host your code repository.
2. **Cloud Host**: [Render.com](https://render.com) or [Railway.app](https://railway.app) (Free / low-cost web service hosting).
3. **Managed PostgreSQL**: [Supabase](https://supabase.com) or [Neon.tech](https://neon.tech) (Free tier available).
4. **Cloud Storage**: [Cloudinary](https://cloudinary.com) (Free 25GB storage tier).

---

## 2. GitHub Setup

1. Initialize git in your local project folder:
   ```bash
   git init
   git add .
   git commit -m "feat: production-ready DeepTrust AI with camera, PostgreSQL, and Cloudinary"
   ```

2. Create a new repository on GitHub (e.g., `deeptrust-ai`).

3. Push your repository:
   ```bash
   git remote add origin https://github.com/your-username/deeptrust-ai.git
   git branch -M main
   git push -u origin main
   ```

> **Note on Model Weights (Hugging Face)**:
> The heavy 2.44 GB UCF-Net deepfake detection model (`model_best.pth`) is **NOT** stored in the Git repository and is excluded via `.gitignore` (`*.pth`).
> Instead, it is hosted on Hugging Face:
> - **Hugging Face Repository**: `PranjalSeluriyal/Deepfake_Detection`
> - **Model File**: `model_best.pth`
> - The application automatically downloads and caches the model on demand using `huggingface_hub` (`services/model_loader.py`).
> - Audio detection weights (~7MB) and face detector assets (~232KB) remain lightweight and deployable.

## 3. Database Setup (PostgreSQL)

You can use **Supabase** or **Neon**:

### Option A: Supabase (Recommended)
1. Sign in to [Supabase](https://supabase.com) and click **New Project**.
2. Name your project (e.g. `deeptrust-db`) and set a strong database password.
3. Once created, go to **Project Settings** &rarr; **Database**.
4. Under **Connection string**, select **URI** and copy the string:
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.xyz.supabase.co:5432/postgres
   ```

### Option B: Neon.tech
1. Sign in to [Neon.tech](https://neon.tech) and create a database.
2. Copy the **Direct connection** PostgreSQL URI.

---

## 4. Cloud Object Storage Setup (Cloudinary)

1. Sign up at [Cloudinary](https://cloudinary.com).
2. From the **Dashboard**, locate your **API Environment Variable** / **CLOUDINARY_URL**:
   ```
   cloudinary://<api_key>:<api_secret>@<cloud_name>
   ```
3. Copy this URL. All images analyzed through DeepTrust AI will automatically upload here and generate CDN URLs stored in PostgreSQL.

---

## 5. Deploying to Render.com (Step-by-Step)

Render is ideal for deploying Python / Flask web services with Gunicorn:

1. Sign up/Log in to [Render.com](https://render.com).
2. Click **New +** &rarr; **Web Service**.
3. Connect your GitHub repository (`deeptrust-ai`).
4. Configure service settings:
   - **Name**: `deeptrust-ai`
   - **Region**: Choose the region closest to you.
   - **Branch**: `main`
   - **Root Directory**: Leave blank.
   - **Runtime**: `Python 3` (or `Docker`)
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     gunicorn wsgi:app --workers 2 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT
     ```
   - **Plan**: Starter or Free tier.

5. Add **Environment Variables** under the **Environment** tab:
   | Key | Value | Description |
   |---|---|---|
   | `FLASK_ENV` | `production` | Enables production mode |
   | `SECRET_KEY` | `your-random-secret-key-32-chars` | Session/CSRF secret |
   | `DATABASE_URL` | `postgresql://postgres:...` | Your PostgreSQL connection string |
   | `CLOUDINARY_URL` | `cloudinary://api_key:secret@name` | Your Cloudinary connection string |
   | `PYTHONIOENCODING` | `utf-8` | Ensures UTF-8 logging |

6. Click **Deploy Web Service**.
7. Render will build the environment, download dependencies, load models once, initialize database tables, and provide your live URL:
   ```
   https://deeptrust-ai.onrender.com
   ```

---

## 6. Camera Permissions & HTTPS

Modern web browsers (Chrome, Safari, Edge, Firefox) **require HTTPS** to allow access to device cameras via `navigator.mediaDevices.getUserMedia()`.

- Render, Railway, and Fly.io automatically provision **free SSL/TLS certificates (HTTPS)** for your deployed app domain (`https://...`).
- When visiting the website on iPhone or Android, tap **"Use Camera"** &rarr; the browser will show a standard prompt:
  > *"deeptrust-ai.onrender.com wants to use your camera" &rarr; Tap **Allow**.*
- You can switch between front and rear cameras using the rotate camera button on mobile.

---

## 7. Verifying the Deployment

Once deployed:

1. **Health Check**:
   Visit: `https://your-domain.com/api/health`
   Expected response:
   ```json
   {
     "status": "healthy",
     "database": "healthy",
     "models": {
       "image_model": "loaded",
       "audio_model": "loaded"
     },
     "storage": "cloudinary"
   }
   ```

2. **Image Upload Test**:
   - Open your site on desktop or mobile.
   - Upload a test photo (JPG, PNG, WebP).
   - Click **Analyze Image**.
   - Verify the forensic verdict ("REAL" or "DEEPFAKE"), confidence score, and metrics appear.

3. **Camera Test**:
   - Tap **Use Camera**.
   - Allow camera permissions.
   - Tap **Capture Photo** &rarr; Review photo &rarr; Tap **Analyze Captured Photo**.

4. **Audit History Check**:
   - Scroll down to the **Audit Ledger** section or visit `/history`.
   - Verify the scan is recorded with its Cloudinary thumbnail and timestamp.

---

## 8. Common Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| **Camera not opening on phone** | Accessed via plain HTTP or permission denied | Ensure the URL begins with `https://`. In browser settings, check Site Settings &rarr; Camera &rarr; Allow. |
| **`sqlalchemy.exc.NoSuchModuleError: postgres`** | Connection string begins with `postgres://` instead of `postgresql://` | Handled automatically by `config.py`. Verify `DATABASE_URL` format. |
| **Application timeout during start** | Model download/loading taking longer than 60s | The start command specifies `--timeout 120`. Models load once during startup. |
| **Images not appearing in cloud storage** | Missing or invalid `CLOUDINARY_URL` | Check your Cloudinary Dashboard credentials and ensure `CLOUDINARY_URL` is set in the cloud host's Environment Variables. |

---

## 9. Vercel Deployment & Architecture Analysis

### Architectural Feasibility for 2.44 GB ML Model

| Constraint | Vercel Serverless Python Functions | Container Platform (Render / Railway / Docker) |
|---|---|---|
| **Max Uncompressed Bundle Size** | 250 MB | Unlimited (typically 10 GB - 50 GB) |
| **Ephemeral Temp Storage (`/tmp`)** | 512 MB &ndash; 1,024 MB | Persistent or container disk (10 GB &ndash; 50 GB) |
| **RAM Memory Limit** | 1,024 MB (Free) &ndash; 3,008 MB (Pro) | 4,096 MB &ndash; 16,384 MB |
| **Max Execution / Cold-Start Timeout** | 10s (Free) &ndash; 60s/300s (Pro) | 300s &ndash; unlimited worker lifetime |
| **UCF-Net 2.44 GB Model Feasibility** | ❌ **UNSUITABLE for ML inference** (exceeds `/tmp` disk and RAM) | ✅ **PRODUCTION-READY** (runs smoothly in container) |

### Recommended Deployment Strategy:

#### Option A: Unified Container Deployment (Recommended)
Deploy the entire application (Flask web frontend + PyTorch ML backend) to a containerized host like **Render**, **Railway**, **AWS ECS / App Runner**, or **Google Cloud Run** (configured with >= 4 GB RAM).
- Single deployment, single service.
- Hugging Face automatically streams and caches the 2.44 GB model on container cold-boot.
- Multi-threaded Gunicorn worker (`--workers 1 --threads 4 --timeout 300`) keeps memory consumption optimal.

#### Option B: Hybrid Architecture (Vercel Frontend + Dedicated ML Backend)
1. **Frontend / Gateway on Vercel**:
   - Deploy this repository to Vercel (using the provided `vercel.json` and `api/index.py`).
   - In Vercel Project Settings &rarr; Environment Variables, set:
     ```bash
     BACKEND_API_URL=https://your-backend-service.onrender.com
     ```
   - The Vercel deployment serves the UI, static assets, and proxies `/api/predict/image`, `/api/predict/audio` directly to the dedicated backend.
2. **Backend on Render / Railway / GPU Instance**:
   - Runs the Flask app with the full PyTorch inference engine and Hugging Face model loader.

---

## 10. How to Update Your Deployed App

Whenever you make changes to the code:
```bash
git add .
git commit -m "feat: hugging face model integration"
git push origin main
```
Your hosting platform will automatically build and redeploy.

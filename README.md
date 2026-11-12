# Leafwatch

Leafwatch is a mobile-ready pest detection website for Botany students. Upload a leaf photo to identify visible pest species (aphids, whiteflies, caterpillars, and other IP102 species), confidence scores, bounding boxes, potential crop damage, and scouting advice.

## Free Hosting Options (Shareable Client Link)

### Option 1: Render Free Web Service (All-in-One: Model + Website)

`serve.py` already serves both the trained model API and the full website (`dist/`), so you only need one free Render service and one URL to share with your client.

1. Push this `leafwatch` folder to a GitHub repository.
2. Sign in to [Render](https://render.com) with GitHub, click **New + → Blueprint**, and select your repository.
3. Render automatically reads `render.yaml` and `Dockerfile` and deploys the service for free (no credit card required).
4. Once `/api/status` shows `"ready": true`, share your Render URL directly with the client:
   ```text
   https://leafwatch-inference.onrender.com
   ```

> **Using your own trained model weights without Hugging Face:** Upload your `best.pt` (or `best.pth`) to a GitHub Release and set `MODEL_URL=https://github.com/<user>/<repo>/releases/download/<tag>/best.pt` in Render's Environment Variables.

---

### Option 2: Vercel (Client Link) + Render (Model Backend)

If you want a `*.vercel.app` link for the client:

1. Deploy the backend on Render (Option 1 above) or Koyeb (`app.koyeb.com`, also 512 MB free Docker tier).
2. Import the same GitHub repository on [Vercel](https://vercel.com/new).
3. In Vercel **Project Settings → Environment Variables**, add:
   ```text
   INFERENCE_API_URL=https://leafwatch-inference.onrender.com/api/scan
   INFERENCE_STATUS_URL=https://leafwatch-inference.onrender.com/api/status
   ```
4. Redeploy on Vercel and share your `https://<your-project>.vercel.app` link.

---

### Option 3: Instant Free Demo Link from Your PC (Zero Cloud Signup)

For a quick live client demo directly from your laptop (no 512 MB RAM limit, zero cloud setup):

1. Start Leafwatch locally:
   ```bash
   python serve.py
   ```
2. In a second terminal, expose port `8000` using Cloudflare's free Quick Tunnel (no account required):
   ```bash
   npx cloudflared tunnel --url http://127.0.0.1:8000
   ```
3. Share the generated `https://*.trycloudflare.com` URL with your client (stays live as long as both terminals are open).

---

## Run Locally

```bash
pip install -r requirements.txt
python serve.py
```

Open `http://127.0.0.1:8000` in your browser. By default `serve.py` loads `model-source/best.pt` (or `training-runs/mobilenet-ip102/best.pth` if present). You can also select a specific checkpoint:

```bash
MODEL_PATH=training-runs/mobilenet-ip102/best.pth python serve.py
```

## Model Training

- `prepare_ip102.py` converts the official IP102 detection archives into `aphid`, `whitefly`, and `caterpillar` classes.
- `train.py` audits splits, trains YOLO, and writes held-out metrics to `evaluation.json`.
- `mobile_detector.py` trains and runs the lightweight MobileNetV3 SSDLite checkpoint (`best.pth`).
- See [training/README.md](training/README.md) for training instructions.

## What Is Kept Where

| Location | Contents |
| --- | --- |
| GitHub + Render / Koyeb | Website (`dist/`), FastAPI service (`serve.py`), and `Dockerfile` |
| Optional Vercel site | Static website (`dist/`) and `/api/*` serverless proxy (`api/scan.js`, `api/status.js`) |
| Training machine | Temporary IP102 images and labels (`training-data/`) |
| Client browser | Local field-log entries (`localStorage`) |

Photos are processed in memory and are not written to disk.

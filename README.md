# 🌿 Leafwatch · Smart Pest Detector

Leafwatch is an AI-powered agricultural pest detection web application developed for the Department of Botany. Users capture or upload a leaf photo to instantly detect visible crop pests (aphids, whiteflies, caterpillars, and other agricultural pests), review bounding boxes and confidence scores, evaluate potential crop damage, and receive actionable integrated pest management (IPM) scouting guidance.

---

## 🚀 Free Hosting on Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is 100% free, provides 1 GB of RAM (ideal for PyTorch & YOLO), supports live mobile camera capture, and gives you a permanent shareable URL (`https://<your-app-name>.streamlit.app`) to send to your client.

### Step-by-Step Deployment:

1. **Commit and Push to GitHub**:
   Make sure all files (`streamlit_app.py`, `app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml`) are in the root of your GitHub repository (`Saqib4746/LeafWatch`):
   ```bash
   git add .
   git commit -m "Deploy Leafwatch to Streamlit Cloud"
   git push origin main
   ```

2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your **GitHub** account.
   - Click **Create app** (or **New app**).
   - Configure the deployment settings:
     - **Repository**: `Saqib4746/LeafWatch`
     - **Branch**: `main`
     - **Main file path**: `streamlit_app.py`
     - **App URL** (optional custom subdomain): e.g. `leafwatch-botany.streamlit.app`
   - Click **Deploy!**

3. **Share with Client**:
   - Streamlit will install the requirements and launch the application in ~2 minutes.
   - Share your live link (`https://<your-app-name>.streamlit.app`) with the client. It works seamlessly on phones, tablets, and desktops!

---

## 💻 Running Locally

### Streamlit Web App:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser.

### Alternative FastAPI Service:
```bash
python serve.py
```
Open `http://127.0.0.1:8000` in your browser.

---

## 🔬 Model & Architecture

- **Default Model**: YOLO11s trained on the IP102 agricultural benchmark (automatically loaded/downloaded).
- **Lightweight Alternative**: MobileNetV3 SSDLite (`training-runs/mobilenet-ip102/best.pth`).
- **Pest Classes**:
  - 🟠 **Aphids** (*Aphidoidea*): green bugs, toxoptera, therioaphis, etc.
  - 🔵 **Whiteflies** (*Aleyrodidae*): sweetpotato whitefly, greenhouse whitefly, etc.
  - 🟢 **Caterpillars** (*Lepidoptera* larvae): armyworms, cutworms, leaf rollers, borers, budworms.
  - 🟣 **Other Field Pests**: Wider IP102 species (thrips, plant hoppers, beetles).

---

## 📂 Project Structure

| File / Folder | Purpose |
| --- | --- |
| `streamlit_app.py` | Main Streamlit web application with detection, camera input, and field log |
| `app.py` | Streamlit entrypoint redirect |
| `packages.txt` | Linux system dependencies (`libgl1`, `libglib2.0-0`) for Streamlit Cloud |
| `requirements.txt` | Python packages for Streamlit, PyTorch, Ultralytics, and Pillow |
| `.streamlit/config.toml` | Custom botany theme styling and server config |
| `mobile_detector.py` | Lightweight MobileNetV3 detector runner |
| `serve.py` | FastAPI inference server (optional alternative) |
| `dist/` | Standalone HTML5/JS frontend (for FastAPI / Vercel) |

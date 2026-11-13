"""Leafwatch · Smart Pest Detection Web Application for Streamlit Community Cloud."""
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import streamlit as st

ROOT = Path(__file__).resolve().parent

# Ensure Ultralytics config is writable (critical on Streamlit Cloud & Linux)
if not os.getenv("YOLO_CONFIG_DIR"):
    local_config = ROOT / ".model-config"
    os.environ["YOLO_CONFIG_DIR"] = str(local_config if local_config.exists() else Path("/tmp/Ultralytics"))

# Configure Streamlit page
st.set_page_config(
    page_title="Leafwatch · Smart Pest Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling matching Leafwatch Department of Botany Theme
st.markdown(
    """
    <style>
    :root {
        --pine: #153e31;
        --deep: #102c23;
        --green: #266548;
        --lime: #d4e8a5;
        --paper: #f6f8f3;
    }
    .main-header {
        background: linear-gradient(135deg, #153e31 0%, #1c523f 60%, #266548 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(21, 62, 49, 0.12);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .main-header p {
        color: #d4e8a5 !important;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 4px;
        margin-bottom: 0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .pest-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .badge-aphid { background-color: #ffe0b2; color: #b7410e; }
    .badge-whitefly { background-color: #e1f5fe; color: #0277bd; }
    .badge-caterpillar { background-color: #e8f5e9; color: #2e7d32; }
    .badge-other { background-color: #f3e5f5; color: #6a1b9a; }
    .stMetric {
        background: #ffffff;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #e0e8e2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Authoritative Pest Information & Scouting Guidance
PEST_GUIDE = {
    "aphid": {
        "name": "Aphids",
        "scientific": "Aphidoidea",
        "color": "#E65100",
        "look": "Small pear-shaped soft-bodied insects, often clustered on fresh shoots, buds, or leaf undersides.",
        "signs": "Curled new leaves, sticky honeydew deposits, stunted growth, and visible ant activity.",
        "damage": "Direct sap feeding weakens shoots. Secreted honeydew fosters dark sooty mold growth.",
        "action": "Inspect shoot tips and leaf undersides. Monitor predator presence (ladybird beetles, lacewings). On sturdy foliage, dislodge localized colonies with a targeted stream of clean water or approved horticultural soap.",
        "source": "https://ipm.ucanr.edu/home-and-landscape/aphids/",
    },
    "whitefly": {
        "name": "Whiteflies",
        "scientific": "Aleyrodidae",
        "color": "#0288D1",
        "look": "Tiny (1-2 mm), powdery white-winged adults that flutter up like a small cloud when disturbed.",
        "signs": "Foliar yellowing (chlorosis), leaf drop, sticky honeydew, and scale-like translucent nymphs on undersides.",
        "damage": "Sap exhaustion weakens the crop; prominent honeydew leads to sooty mold; significant risk of vectoring viral diseases.",
        "action": "Check undersides of mid-canopy leaves. Deploy yellow sticky cards to track adult flight density. In enclosed settings, introduce parasitic wasps (*Encarsia formosa*) or apply targeted insecticidal soaps.",
        "source": "https://ipm.ucanr.edu/home-and-landscape/whiteflies/",
    },
    "caterpillar": {
        "name": "Caterpillars",
        "scientific": "Lepidoptera larvae",
        "color": "#2E7D32",
        "look": "Soft, cylindrical segmented larvae with distinct head capsules and chewing mouthparts, often camouflaged.",
        "signs": "Ragged chewed leaf margins, irregular holes, folded/rolled leaves, and dark granular frass (droppings).",
        "damage": "Rapid defoliation can severely cut photosynthetic capacity. Boring caterpillars may tunnel into stems or fruit.",
        "action": "Scout for fresh feeding notches and frass. For small garden plots, hand-pick and destroy confirmed specimens. For larger plantings, consider targeted microbial bio-pesticides like *Bacillus thuringiensis* (Bt).",
        "source": "https://gardeningsolutions.ifas.ufl.edu/care/pests-and-diseases/pests/integrated-pest-management/",
    },
    "other": {
        "name": "Other field pest",
        "scientific": "IP102 identified species",
        "color": "#7B1FA2",
        "look": "Insect specimen identified within the wider IP102 agricultural benchmark (e.g. thrips, planthoppers, beetles).",
        "signs": "Examine leaf surface for stippling, puncture marks, or skeletonization.",
        "damage": "Damage varies by species and host crop. Review the exact species name identified by the model.",
        "action": "Cross-reference the detected species with local agricultural extension materials before applying treatment.",
        "source": "https://www.fao.org/plant-production-protection/integrated-pest-management/en/",
    },
}


def group_pest(label: str) -> str:
    """Map IP102 species labels to the four standard Leafwatch advice groups."""
    name = " ".join(label.lower().split())
    aphid_terms = ("aphid", "aphis", "green bug", "therioaphis", "toxoptera")
    if any(term in name for term in aphid_terms):
        return "aphid"
    if "aleuro" in name or "trialeurodes" in name or "whitefly" in name:
        return "whitefly"
    caterpillar_terms = (
        "caterpillar", "army worm", "cutworm", "borer",
        "leaf roller", "budworm", "moth", "prodenia litura",
    )
    if any(term in name for term in caterpillar_terms):
        return "caterpillar"
    return "other"


@st.cache_resource(show_spinner="Loading IP102 pest detection model...")
def load_detector():
    """Load and cache the trained IP102 object detector."""
    import torch
    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))

    # Check local weights first
    model_path = os.getenv("MODEL_PATH", "").strip()
    if model_path:
        p = Path(model_path)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            if p.suffix.lower() == ".pth":
                from mobile_detector import MobileDetector
                return {"type": "mobile", "model": MobileDetector(p), "name": "MobileNetV3 SSDLite (IP102)"}
            from ultralytics import YOLO
            return {"type": "yolo", "model": YOLO(str(p), task="detect"), "name": "YOLO11s (Local weights)"}

    # Check common local candidate paths
    for cand in (
        ROOT / "model-source" / "best.pt",
        ROOT / "models" / "best.pt",
        ROOT / "best.pt",
    ):
        if cand.is_file():
            from ultralytics import YOLO
            return {"type": "yolo", "model": YOLO(str(cand), task="detect"), "name": "YOLO11s (Local weights)"}

    for cand in (
        ROOT / "training-runs" / "mobilenet-ip102" / "best.pth",
        ROOT / "models" / "best.pth",
    ):
        if cand.is_file():
            from mobile_detector import MobileDetector
            return {"type": "mobile", "model": MobileDetector(cand), "name": "MobileNetV3 SSDLite (Local weights)"}

    # Fall back to public Hugging Face download (automatic on cloud)
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    weights = hf_hub_download(repo_id="underdogquality/yolo11s-pest-detection", filename="best.pt")
    return {"type": "yolo", "model": YOLO(weights, task="detect"), "name": "YOLO11s (IP102 Public Weights)"}


def annotate_image(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes and confidence pills onto the leaf image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    # Try loading a readable default font
    try:
        font = ImageFont.truetype("arial.ttf", size=max(13, int(min(width, height) * 0.024)))
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        box = det["box"]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int((box[0] + box[2]) * width)
        y2 = int((box[1] + box[3]) * height)

        group = det["pest"]
        hex_color = PEST_GUIDE.get(group, PEST_GUIDE["other"])["color"]

        # Convert hex to RGB
        rgb_color = tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

        # Draw outer rectangle
        line_w = max(2, int(min(width, height) * 0.004))
        for offset in range(line_w):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=rgb_color)

        # Draw label pill
        label_text = f"{det['species']} {int(det['confidence'] * 100)}%"
        bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pill_y1 = max(0, y1 - text_h - 6)
        pill_y2 = pill_y1 + text_h + 6

        draw.rectangle([x1, pill_y1, x1 + text_w + 10, pill_y2], fill=rgb_color)
        draw.text((x1 + 5, pill_y1 + 2), label_text, fill=(255, 255, 255), font=font)

    return annotated


def run_inference(detector_info, image: Image.Image, conf_threshold: float = 0.35) -> list:
    """Run model inference and return formatted detections."""
    # Resize down if overly massive to save memory
    proc_img = ImageOps.exif_transpose(image).convert("RGB")
    if max(proc_img.size) > 1280:
        proc_img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)

    m_type = detector_info["type"]
    model = detector_info["model"]

    detections = []
    if m_type == "mobile":
        raw = model.detect(proc_img)
        for item in raw:
            if item["confidence"] >= conf_threshold:
                detections.append({
                    "pest": item["pest"],
                    "species": item["pest"],
                    "confidence": round(float(item["confidence"]), 4),
                    "box": item["box"],
                })
    else:
        result = model.predict(proc_img, conf=conf_threshold, imgsz=640, max_det=150, verbose=False)[0]
        boxes = result.boxes.xyxyn.tolist()
        scores = result.boxes.conf.tolist()
        labels = result.boxes.cls.int().tolist()

        for (x1, y1, x2, y2), conf, lbl in zip(boxes, scores, labels):
            x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
            x2, y2 = min(1.0, float(x2)), min(1.0, float(y2))
            w = max(0.0, min(1.0 - x1, x2 - x1))
            h = max(0.0, min(1.0 - y1, y2 - y1))
            if w > 0 and h > 0:
                species = str(result.names[lbl])
                detections.append({
                    "pest": group_pest(species),
                    "species": species,
                    "confidence": round(float(conf), 4),
                    "box": [round(x1, 5), round(y1, 5), round(w, 5), round(h, 5)],
                })
    return detections


# Initialize Session State
if "observations" not in st.session_state:
    st.session_state.observations = []

# Sidebar Navigation & Settings
with st.sidebar:
    st.image(str(ROOT / "dist" / "leaf.jpg") if (ROOT / "dist" / "leaf.jpg").is_file() else "🌿", use_container_width=True)
    st.markdown("### 🌿 Leafwatch Scout")
    st.caption("Department of Botany · Smart Pest Detector")

    st.divider()
    st.markdown("**Detection Parameters**")
    conf_slider = st.slider("Confidence Threshold", min_value=0.15, max_value=0.85, value=0.35, step=0.05)

    st.divider()
    # Model status badge
    try:
        det_info = load_detector()
        st.success(f"🟢 **Model Ready**\n\n`{det_info['name']}`")
    except Exception as exc:
        st.error(f"🔴 **Model Error**: {exc}")
        det_info = None

    st.divider()
    st.markdown("**About Leafwatch**")
    st.caption("Leafwatch assists students and agronomists in rapid scouting of crop leaves for early pest detection.")

# Main Application Banner
st.markdown(
    """
    <div class="main-header">
        <p>Department of Botany · Field Scouting Project</p>
        <h1>🌿 Leafwatch · Smart Pest Detector</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_overview, tab_scan, tab_log, tab_guide = st.tabs([
    "📊 Overview",
    "🔍 Scan Leaf",
    "📋 Field Log",
    "📖 Pest Guide",
])

# -------------------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------------------
with tab_overview:
    obs = st.session_state.observations
    alerts_count = sum(1 for o in obs if len(o["detections"]) > 0)
    fields_count = len(set(o["field"].lower() for o in obs)) if obs else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations", len(obs), help="Scans recorded in this session")
    col2.metric("Pest Alerts", alerts_count, help="Observations with confirmed insect detections")
    col3.metric("Plots Monitored", fields_count, help="Unique plots or fields recorded")

    st.markdown("---")
    st.subheader("Recent Observations")
    if not obs:
        st.info("No field scans recorded yet. Go to **🔍 Scan Leaf** to analyze your first leaf photo!")
    else:
        for item in reversed(obs[-5:]):
            n_pests = len(item["detections"])
            badge_class = "badge-caterpillar" if n_pests > 0 else "badge-other"
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #e0e8e2; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 1.1rem; color: #153e31;">{item['field']}</strong>
                            <p style="margin: 3px 0 0; color: #5e7065; font-size: 0.88rem;">{item['timestamp']}</p>
                        </div>
                        <span class="pest-badge {badge_class}">
                            {n_pests} pest{'s' if n_pests != 1 else ''} detected
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -------------------------------------------------------------
# TAB 2: SCAN LEAF
# -------------------------------------------------------------
with tab_scan:
    st.subheader("Capture or Upload Leaf Photo")
    st.write("Ensure the leaf is in focus and well-lit. Both upper and lower leaf surfaces can harbor insect colonies.")

    col_input, col_display = st.columns([1, 1], gap="medium")

    with col_input:
        field_name = st.text_input("Plot or Field Identifier", value="Botany Garden · Plot A", max_chars=80)
        input_mode = st.radio("Photo Input Source", ["Upload Image File", "Use Device Camera"], horizontal=True)

        uploaded_file = None
        if input_mode == "Upload Image File":
            uploaded_file = st.file_uploader(
                "Choose a leaf photo",
                type=["jpg", "jpeg", "png", "webp"],
                help="Accepts JPG, PNG, or WebP up to 15MB",
            )
        else:
            uploaded_file = st.camera_input("Take a photo of the leaf")

        analyze_clicked = st.button("🚀 Analyze Photo", type="primary", use_container_width=True, disabled=uploaded_file is None)

    with col_display:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Specimen", use_container_width=True)

            if analyze_clicked:
                if det_info is None:
                    st.error("Detector is unavailable. Please check model logs.")
                else:
                    with st.spinner("Analyzing leaf for insect pests..."):
                        detections = run_inference(det_info, image, conf_threshold=conf_slider)

                    annotated = annotate_image(image, detections)
                    st.image(annotated, caption="Detection Results with Bounding Boxes", use_container_width=True)

                    # Save to session log
                    log_entry = {
                        "id": len(st.session_state.observations) + 1,
                        "field": field_name.strip() or "Unnamed Plot",
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "detections": detections,
                        "count": len(detections),
                    }
                    st.session_state.observations.append(log_entry)

                    st.markdown("---")
                    if not detections:
                        st.success("✅ **No pests detected** above the confidence threshold. Foliage appears clear of heavy infestation.")
                    else:
                        st.warning(f"⚠️ **{len(detections)} Pest Detection(s) Identified:**")

                        # Group by category
                        grouped_hits = {}
                        for d in detections:
                            grouped_hits.setdefault(d["pest"], []).append(d)

                        for group, hits in grouped_hits.items():
                            info = PEST_GUIDE[group]
                            conf_strs = [f"{int(h['confidence'] * 100)}%" for h in hits]
                            species_set = sorted(set(h["species"] for h in hits))
                            with st.expander(f"**{info['name']}** ({len(hits)} instances detected)", expanded=True):
                                st.markdown(f"**Scientific Group:** *{info['scientific']}*")
                                st.markdown(f"**Identified Species:** {', '.join(species_set)}")
                                st.markdown(f"**Confidence Levels:** {', '.join(conf_strs)}")
                                st.markdown(f"**Potential Crop Damage:** {info['damage']}")
                                st.markdown(f"**Scouting & Action:** {info['action']}")
                                st.markdown(f"[🔗 Official IPM Management Guide]({info['source']})")

# -------------------------------------------------------------
# TAB 3: FIELD LOG
# -------------------------------------------------------------
with tab_log:
    st.subheader("Field Observation History")
    obs = st.session_state.observations

    if not obs:
        st.info("No field observations recorded in this session yet.")
    else:
        # Build DataFrame for display and download
        rows = []
        for o in obs:
            counts = {"aphid": 0, "whitefly": 0, "caterpillar": 0, "other": 0}
            for d in o["detections"]:
                counts[d["pest"]] = counts.get(d["pest"], 0) + 1

            rows.append({
                "ID": o["ID"] if "ID" in o else o["id"],
                "Field / Plot": o["field"],
                "Timestamp": o["timestamp"],
                "Total Pests": o["count"],
                "Aphids": counts["aphid"],
                "Whiteflies": counts["whitefly"],
                "Caterpillars": counts["caterpillar"],
                "Other": counts["other"],
            })

        df = pd.DataFrame(rows)

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            search_query = st.text_input("🔍 Search observations by plot name", "")
        with col_f2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"leafwatch_field_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if search_query:
            filtered_df = df[df["Field / Plot"].str.contains(search_query, case=False, na=False)]
        else:
            filtered_df = df

        st.dataframe(filtered_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: PEST GUIDE
# -------------------------------------------------------------
with tab_guide:
    st.subheader("Field Identification & IPM Guide")
    st.write("Reference guide for field scouting and diagnosis of common leaf pests.")

    for pest_key, guide_info in PEST_GUIDE.items():
        with st.container():
            st.markdown(
                f"""
                <div style="background: white; border-left: 5px solid {guide_info['color']}; border-radius: 8px; padding: 18px; margin-bottom: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
                    <h3 style="margin-top: 0; color: #153e31;">{guide_info['name']} <small style="color: #5e7065; font-size: 0.9rem;">({guide_info['scientific']})</small></h3>
                    <p><strong>Visual Identification:</strong> {guide_info['look']}</p>
                    <p><strong>Common Signs:</strong> {guide_info['signs']}</p>
                    <p><strong>Potential Crop Damage:</strong> {guide_info['damage']}</p>
                    <p><strong>Recommended Action:</strong> {guide_info['action']}</p>
                    <a href="{guide_info['source']}" target="_blank" style="color: #266548; font-weight: 700; text-decoration: none;">View Official Management Guide ↗</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("📷 Field Specimen Photography Tips")
    st.markdown(
        """
        - **Lighting**: Capture photos in diffuse, even daylight. Avoid direct harsh sunlight glare.
        - **Focus**: Tap your phone screen to lock focus directly on the leaf blade or insect colony.
        - **Both Sides**: Always examine the leaf underside, where aphids, whitefly scales, and young caterpillars congregate.
        - **Multiple Angles**: If damage is severe, take a wide shot of the affected foliage followed by a macro close-up.
        """
    )

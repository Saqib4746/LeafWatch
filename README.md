# Leafwatch

Leafwatch is a mobile-ready pest detection website for Botany students. Students upload a leaf photo through one Vercel link. The site returns the detected pest species, confidence, potential crop damage, and practical scouting advice.

## Deploy the website on GitHub, Render, and Vercel

1. Create a GitHub repository and upload this `leafwatch` folder.
2. Sign in to Render with GitHub, select **New +**, then **Blueprint**, and select the repository. Render detects `render.yaml` and deploys the detector.
3. Copy the Render service URL after its `/api/status` page says `"ready": true`.
4. Sign in to Vercel with GitHub and import the same repository.
5. In Vercel Project Settings, add the two environment variables below, using your Render URL, then deploy again.

Students only receive the Vercel URL. They never see model credentials, training images, or the separate inference endpoint URL.

## Copy these two values into Vercel

Vercel hosts the website and lightweight API routes. The trained YOLO model runs on an inference service because Vercel serverless functions do not support full PyTorch model execution reliably.

```text
INFERENCE_API_URL=https://your-render-service.onrender.com/api/scan
INFERENCE_STATUS_URL=https://your-render-service.onrender.com/api/status
```

`api/scan.js` forwards each photo to the model service. `api/status.js` lets the interface show whether the model is ready. No Hugging Face account is required. Render downloads the public IP102 model automatically when it starts.

## Model training

The source files are included in this repository:

- `prepare_ip102.py` converts the official IP102 detection material into aphid, whitefly and caterpillar classes.
- `train.py` audits the splits, trains YOLO, and writes held-out test metrics to `evaluation.json`.
- `serve.py` is the inference service that loads the model and exposes the contract expected by Vercel.

Download the IP102 resources through the [official project source](https://github.com/xpwu95/IP102). Keep the dataset outside this GitHub repository. The model service can begin with the public [YOLO11s model trained on IP102](https://huggingface.co/underdogquality/yolo11s-pest-detection), then later use the project’s own `best.pt` after training.

Run the model training on Google Colab, Kaggle, a university GPU machine, or another GPU-enabled host. Use the process in [training/README.md](training/README.md). Training output should be evaluated on independent field photos before any accuracy claim is made.

## What is kept where

| Location | Contents |
| --- | --- |
| GitHub and Vercel | Website source, Vercel API proxy, and training code |
| Training machine | Temporary IP102 images and labels |
| Inference service | Model weights and short-lived photo processing |
| Student browser | Their local field-log entries |

Photos are sent to the inference service for analysis. Leafwatch processes them in memory and does not write them to disk.

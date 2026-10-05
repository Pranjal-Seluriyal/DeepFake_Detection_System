# UCF-Net Model Weights

The trained UCF-Net model weights (`model_best.pth`, ~2.44 GB) are hosted on Hugging Face:
- **Repository**: [PranjalSeluriyal/Deepfake_Detection](https://huggingface.co/PranjalSeluriyal/Deepfake_Detection)
- **Model File**: `model_best.pth`

### Automatic Download & Caching
The application uses `services/model_loader.py` to automatically fetch and cache `model_best.pth` from Hugging Face on demand using `huggingface_hub.hf_hub_download`.

The model weights are excluded from Git commits via `.gitignore` (`*.pth`).

To configure a custom model or private Hugging Face repo:
```bash
HF_MODEL_REPO=PranjalSeluriyal/Deepfake_Detection
HF_MODEL_FILENAME=model_best.pth
HF_TOKEN=your_huggingface_token_if_private
```

import gradio as gr
from pathlib import Path
import json
import time
from functools import partial

import torch
import cv2
import numpy as np
from safetensors.torch import load_file

from src.data_utils import read_video
from src.models import ConvNeXtTransformer
from src.dataset import VideoAugmentation

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models/convnext-transformer"
LABEL_MAPPING_PATH = ROOT / "data/dataset/label_mapping.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FRAMES = 16
MAX_FRAMES = 32


def get_most_recent_file(directory):
    files = [f for f in directory.iterdir() if f.is_file()]
    most_recent = max(files)
    return most_recent


def load_model(model_path, num_classes):
    model = ConvNeXtTransformer(num_classes=num_classes)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)


def resample_frames(frames, target_frames):
    total = frames.shape[0]

    if total >= target_frames:
        indices = torch.linspace(0, total - 1, target_frames).long()
    else:
        indices = torch.arange(total)
        pad = target_frames - total
        indices = torch.cat([indices, indices[-1].repeat(pad)])

    return frames[indices]


def normalize_frames(frames, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    frames = frames.permute(0, 3, 1, 2).float() / 255.0

    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)

    return (frames - mean) / std


def preprocess_video(video_path):
    transforms = VideoAugmentation(mode="test")
    frames = read_video(video_path)  # (T,H,W,C) uint8

    frames = transforms(frames)
    frames = resample_frames(frames, target_frames=TARGET_FRAMES)
    frames = normalize_frames(frames)

    return frames.unsqueeze(0)  # (1, T, C, H, W)


def predict_video(video_path, model, id2label):
    if video_path is None:
        return "No video"

    video_tensor = preprocess_video(video_path).to(DEVICE)

    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)

    label = id2label[int(pred.item())]
    confidence = float(conf.item())

    return f"[RESULT] {label}\nCONFIDENCE: {confidence:.2%}"


def predict_webcam(frame, state, model, id2label):
    if state is None:
        state = {"buffer": [], "cooldown": 0, "log": []}
        
    timestamp = time.strftime("%H:%M:%S")

    buffer = state["buffer"]

    # preprocess frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).unsqueeze(0)
    frame = VideoAugmentation(mode="test")(frame)
    frame = normalize_frames(frame)[0]

    # cooldown
    if state["cooldown"] > 0:
        state["cooldown"] -= 1
        status = f"⏳ {timestamp} - [SYSTEM] Waiting..."
        
        log_text = "\n".join(state["log"])
        return f"{status}\n\n{log_text}", state

    buffer.append(frame)

    # Collecting frames
    if len(buffer) < MAX_FRAMES:
        status = f"📥 {timestamp} - [SYSTEM] Collecting {len(buffer)}/{MAX_FRAMES}..."
        
        log_text = "\n".join(state["log"])
        return f"{status}\n\n{log_text}", state

    # đủ frame → predict
    frames = torch.stack(buffer)
    frames = resample_frames(frames, target_frames=TARGET_FRAMES)

    x = frames.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        
    label = id2label[int(pred.item())]
    confidence = float(conf.item())

    result = f"✅ {timestamp} - [RESULT] {label}\n" + \
                   " " * 30 + f"CONFIDENCE: {confidence:.2%}\n"

    # lưu log
    state["log"].append(result)
    
    # limit log (tránh dài quá)
    if len(state["log"]) > 10:
        state["log"].pop(0)

    # reset
    state["buffer"] = []
    state["cooldown"] = 15

    status = f"✅ {timestamp} - [SYSTEM] Done"

    log_text = "\n".join(state["log"])
    return f"{status}\n\n{log_text}", state


if __name__ == "__main__":
    model_path = get_most_recent_file(MODEL_DIR)
    
    example_video_paths = [
        [str(ROOT / path)] for path in [
            "data/dataset/train/Ăn/148050.mp4",
            "data/dataset/train/Bộ y tế/141672.mp4",
            "data/dataset/train/Chạy/100425.mp4"
        ]
    ]
    
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    
    model = load_model(model_path, num_classes=len(id2label))
    
    with gr.Blocks() as demo:
        gr.Markdown("# Sign Language Recognition Demo")
        
        with gr.Tab("Upload Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    video_input = gr.Video(sources="upload", label="📹 Input Video")
                    
                    btn = gr.Button("🚀 Predict", variant="primary")
                    
                with gr.Column(scale=1):
                    output = gr.Textbox(
                        value="Result will appear here...",
                        label="Prediction"
                    )
                    
                # with gr.Column(scale=1):
                    gr.Markdown("### 📂 Examples")
                    gr.Examples(
                        examples=example_video_paths,
                        inputs=video_input
                    )
                    
                btn.click(
                    partial(predict_video, model=model, id2label=id2label),
                    inputs=video_input,
                    outputs=output
                )

        with gr.Tab("Webcam"):
            with gr.Row():
                webcam = gr.Image(
                    sources="webcam",
                    streaming=True,
                    label="Webcam",
                    #scale=2,
                    elem_id="webcam",
                    width=250
                )
                state = gr.State(None)
                out = gr.Textbox(
                    label="TRANSCRIPTION LOG",
                    lines=10,
                    # scale=1,
                    autoscroll=True
                )

                webcam.stream(
                    partial(predict_webcam, model=model, id2label=id2label),
                    inputs=[webcam, state],
                    outputs=[out, state],
                    stream_every=0.1,
                    concurrency_limit=1
                )
        
    demo.launch(css="""
    #webcam video {
        aspect-ratio: 1 / 1;
        object-fit: cover;
    }
    """)
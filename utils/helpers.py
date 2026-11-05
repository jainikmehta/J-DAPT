import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import numpy as np
import os
import random
import time
from tqdm import tqdm

from openai import OpenAI
from PIL import Image
import torch
import torch.nn as nn

from .params import *


def get_llm_client():
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        key_path = os.path.expanduser(llm_api_key_path)
        if os.path.exists(key_path):
            with open(key_path) as f:
                api_key = f.read().strip()
    return OpenAI(base_url=llm_base_url, api_key=api_key or "EMPTY")


def encode_image_data_url(image_path):
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}" if ext in ["png", "webp"] else "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_vlm_chat(prompt, images, model=None):
    if model is None:
        model = vlm_model
    client = get_llm_client()
    content = []
    for img_path in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_data_url(img_path)}
        })
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=500
    )
    return response.choices[0].message.content


def setSeed(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def imgs2mp4(images, path, fps=12):
    import cv2
    if not os.path.exists(path):
        frame = cv2.imread(images[0])
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for frame_name in images:
            frame = cv2.imread(frame_name)
            video.write(frame)
        video.release()


def extract_frame(video_path, frame_index=0):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = frame_index % total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not extract frame {frame_index} from {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def goal_target_prompt(prompt, images, path, model=None):
    content = call_vlm_chat(prompt, images, model=model)
    lines = content.splitlines()
    goal_line = next((line for line in lines if line.lower().startswith("goal:") or "goal:" in line.lower()), "")
    target_line = next((line for line in lines if line.lower().startswith("target:") or "target:" in line.lower()), "")

    goal = goal_line.removeprefix("GOAL:").strip()
    target = target_line.removeprefix("TARGET:").strip()

    if not goal or not target:
        return
    
    # Saving text
    with open(os.path.join(path), 'w') as f:
        f.write(f"GOAL: {goal}\n")
        f.write(f"TARGET: {target}\n")
        f.write(f"IMAGES: {', '.join(images)}\n")


def benign_prompt(prompt, images, path, model=None):
    content = call_vlm_chat(prompt, images, model=model)
    lines = content.splitlines()
    query_line = next((line for line in lines if line.lower().startswith("query:")), "")

    query = query_line.removeprefix("QUERY:").strip()

    if not query:
        return
    
    # Saving text
    with open(os.path.join(path), 'w') as f:
        f.write(f"QUERY: {query}\n")
        f.write(f"IMAGES: {', '.join(images)}\n")


def generate_embedding(image_path, text, model, processor):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)

    text_embeds = outputs.text_embeds.to(device)
    image_embeds = outputs.image_embeds.to(device)

    return text_embeds, image_embeds

def generate_dinotxt_embedding(image_path, text, model, tokenizer):
    from dinov2.data.transforms import make_classification_eval_transform
    # Image
    image_preprocess = make_classification_eval_transform()
    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.stack([image_preprocess(image)], dim=0).cuda()

    # Text
    tokenized_texts_tensor = tokenizer.tokenize(text).cuda()

    # Embedding
    with torch.autocast('cuda', dtype=torch.float):
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(tokenized_texts_tensor)

    return text_features, image_features


@contextmanager
def status(msg):
    print(f"[⏱️] {msg}...", end="", flush=True)
    start = time.time()
    try:
        yield
        print(f" Done ✅ ({time.time() - start:.1f}s)")
    except Exception as e:
        print(f" Failed ❌ ({e})")
        raise


def get_embedding(type, embeddings_folder):
    embeddings = []
    walked_files = list(os.walk(os.path.join(embeddings_folder, type)))
    for root, _, files in walked_files:
        for file in tqdm(files, desc=f'{root.split("/")[-1]}'):
            if file.endswith('.npz'):
                embeddings.append(np.load(os.path.join(root, file)))
    return embeddings


def load_embeddings(embeddings, domain_label, label='all', mode='full'):
    """
    Load embeddings and return concatenated vectors along with labels and domain labels.

    Parameters:
        embeddings (list): List of datasets to process.
        domain_label (str/int): Label to assign to all data in this domain.
        label (str): Filter for 'all', 'jb', 'rt', or 'bq'.
        mode (str): 
            - 'full': text + image + fused
            - 'nofused': text + image
            - 'text': only text

    Returns:
        X (np.ndarray): Feature vectors.
        y (np.ndarray): Labels (0 = jailbreak/redteam, 1 = benign).
        domain (np.ndarray): Domain labels.
    """
    X, y, domain = [], [], []

    def get_parts(data, prefix):
        parts = []
        if mode == 'text':
            parts.append(data[f'{prefix}_text'].reshape(-1))
        elif mode == 'nofused':
            parts.extend([
                data[f'{prefix}_text'].reshape(-1),
                data[f'{prefix}_image'].reshape(-1)
            ])
        elif mode == 'full':
            parts.extend([
                data[f'{prefix}_text'].reshape(-1),
                data[f'{prefix}_image'].reshape(-1),
                data[f'{prefix}_fused'].reshape(-1)
            ])
        return np.concatenate(parts)

    for dataset in embeddings:
        for data in dataset:
            if 'jb_fused' in data and label in ['all', 'jb']:
                X.append(get_parts(data, 'jb'))
                y.append(0)
                domain.append(domain_label)

            if 'rt_fused' in data and label in ['all', 'rt']:
                X.append(get_parts(data, 'rt'))
                y.append(0)
                domain.append(domain_label)

            if 'bq_fused' in data and label in ['all', 'bq']:
                X.append(get_parts(data, 'bq'))
                y.append(1)
                domain.append(domain_label)

    # print(X)
    return np.array(X), np.array(y), np.array(domain)


def extract_label_embeddings(embeddings, label):
    jb_embeddings = []
    for dataset in embeddings:
        for data in dataset:
            if f'{label}_text' in data and f'{label}_image' in data:
                jb_embeddings.append(np.concatenate((data[f'{label}_text'], data[f'{label}_image'])).reshape(-1))
            elif f'{label}_fused' in data:
                jb_embeddings.append(data[f'{label}_fused'].reshape(-1))
    return np.array(jb_embeddings)


def compute_coral(X, Y):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Computing covariance matrices
    cov_X = torch.mm((X - X.mean(dim=0)).T, (X - X.mean(dim=0))) / (X.shape[0] - 1)
    cov_Y = torch.mm((Y - Y.mean(dim=0)).T, (Y - Y.mean(dim=0))) / (Y.shape[0] - 1)

    # Computing the Frobenius norm of the covariance difference
    coral_distance = torch.norm(cov_X - cov_Y, p='fro')
    
    return coral_distance.item()


def coral_adaptation(X_source, X_target):
    X_source = torch.tensor(X_source, dtype=torch.float32)
    X_target = torch.tensor(X_target, dtype=torch.float32)

    # Computing source and target covariance matrices
    cov_src = torch.mm((X_source - X_source.mean(dim=0)).T, (X_source - X_source.mean(dim=0))) / (X_source.shape[0] - 1)
    cov_tgt = torch.mm((X_target - X_target.mean(dim=0)).T, (X_target - X_target.mean(dim=0))) / (X_target.shape[0] - 1)

    # Computing whitening transformation for source
    U_src, S_src, V_src = torch.svd(cov_src)
    W_src = torch.mm(U_src, torch.diag(1.0 / torch.sqrt(S_src + 1e-5)))

    # Computing re-coloring transformation using target covariance
    U_tgt, S_tgt, V_tgt = torch.svd(cov_tgt)
    W_tgt = torch.mm(U_tgt, torch.diag(torch.sqrt(S_tgt + 1e-5)))

    # Transforming source embeddings
    X_source_aligned = torch.mm(torch.mm(X_source - X_source.mean(dim=0), W_src), W_tgt) + X_target.mean(dim=0)

    return X_source_aligned.numpy()


class WeightedLoss(nn.Module):
    # Applying weights to loss function in fine-tuning
    def __init__(self, base_loss):
        super(WeightedLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, outputs, targets, weights):
        loss = self.base_loss(outputs, targets)
        return torch.mean(weights * loss)
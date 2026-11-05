import ast
import json
import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm
import warnings

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor, BlipProcessor, BlipModel

from utils.helpers import *
from utils.models import *
from utils.params import *

warnings.filterwarnings("ignore")



def train_cross_attention_model(jb28k_path, vqa_path, model, processor, model_save_path, embed_dim):
    image_embed_list, text_embed_list = [], []

    jb28k_df = pd.read_csv(jb28k_path + '/JailBreakV_28K.csv')
    vqa_df = pd.read_csv(vqa_path + '/data.csv')

    jb_len = len(jb28k_df)
    vqa_len = len(vqa_df)
    cap = min(jb_len, vqa_len)

    for index, row in tqdm(jb28k_df.iloc[:cap].iterrows(), total=cap, desc=f'Loading JB28K ({embed_dim})'):
        jb_type = row['from'].replace(' ', '_')
        image_path = f'{jb28k_path}/{row["image_path"]}'

        if 'dinotxt' not in model_save_path:
            jb_text_embedding, jb_image_embedding = generate_embedding(image_path, row['jailbreak_query'], model, processor)
        else:
            jb_text_embedding, jb_image_embedding = generate_dinotxt_embedding(image_path, row['jailbreak_query'], dinotxt_model, dinotxt_tokenizer)
        text_embed_list.append(torch.tensor(jb_text_embedding))
        image_embed_list.append(torch.tensor(jb_image_embedding))

    for index, row in tqdm(vqa_df.iloc[:cap].iterrows(), total=cap, desc=f'Loading VQA ({embed_dim})'):
        image_path = f'{vqa_path}/images/{row["image_id"]}.png'
        if 'dinotxt' not in model_save_path:
            bq_text_embedding, bq_image_embedding = generate_embedding(image_path, row['question'], model, processor)
        else:
            bq_text_embedding, bq_image_embedding = generate_dinotxt_embedding(image_path, row['question'], dinotxt_model, dinotxt_tokenizer)
        text_embed_list.append(torch.tensor(bq_text_embedding))
        image_embed_list.append(torch.tensor(bq_image_embedding))

    text_embeds = torch.stack(text_embed_list)
    image_embeds = torch.stack(image_embed_list)

    dataset = MultimodalEmbeddingDataset(image_embeds, text_embeds)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    fusion_model = CrossAttentionFusion(embed_dim=embed_dim, num_heads=8, output_dim=embed_dim).to(device)
    optimizer = optim.AdamW(fusion_model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        fusion_model.train()
        total_loss = 0
        for text_batch, image_batch in loader:
            text_batch = text_batch.to(device).float()
            image_batch = image_batch.to(device).float()

            fused = fusion_model(text_batch, image_batch)
            target = 0.5 * (text_batch + image_batch)
            loss = loss_fn(fused, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[💪 EPOCH {epoch+1} | {embed_dim}] Loss: {total_loss / len(loader):.4f}", end='\r')

    torch.save(fusion_model.state_dict(), model_save_path)


def embedding_aboships(aboships_path, model, processor, output='embeddings'):
    # Initializing fusion model
    if output == 'embeddings':
        fusion_model = CrossAttentionFusion().to(device)
    elif output == 'siglip':
        fusion_model = CrossAttentionFusion(embed_dim=768, output_dim=768).to(device).half()
    elif output == 'blip':
        fusion_model = CrossAttentionFusion(embed_dim=512, output_dim=512).to(device)
    elif output == 'align':
        fusion_model = CrossAttentionFusion(embed_dim=640, output_dim=640).to(device)
    elif output == 'florence2' or output == 'openclip':
        fusion_model = CrossAttentionFusion(embed_dim=1024, output_dim=1024).to(device)
    elif output == 'dinotxt':
        fusion_model = CrossAttentionFusion(embed_dim=2048, output_dim=2048).to(device)
    fusion_model.load_state_dict(torch.load(f"./models/fusion_model_{output}.pth", map_location=device))
    fusion_model.eval()
    # Creating embedding path
    results_path = aboships_path.replace('datasets', output).replace('images', '')
    os.makedirs(results_path, exist_ok=True)

    # Listing all videos
    mp4_files = []
    for root, dirs, files in os.walk(aboships_path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))

    for video_path in tqdm(sorted(mp4_files), desc='ABOSHIPS'):
        bq_path = video_path.replace('.mp4', '_bq.txt')
        jb_path = bq_path.replace('_bq.txt', '_jailbreak.json')

        filename = os.path.basename(bq_path)
        save_path = os.path.join(results_path, filename.replace('_bq.txt', '.npz'))

        if not os.path.exists(bq_path) or not os.path.exists(jb_path):
            continue

        # Reading benign queries
        with open(bq_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        bq_query = ""
        image_path = ""
        for line in lines:
            if line.startswith("QUERY: "):
                bq_query = line[len("QUERY: "):].strip()
            if line.startswith("IMAGES: "):
                image_path = line[len("IMAGES: "):].strip().split(',')[0]

        if output != 'dinotxt':
            bq_text_embedding, bq_image_embedding = generate_embedding(image_path, bq_query, model, processor)
        else:
            bq_text_embedding, bq_image_embedding = generate_dinotxt_embedding(image_path, bq_query, dinotxt_model, dinotxt_tokenizer)
        bq_fused_embedding = fusion_model(bq_text_embedding, bq_image_embedding)

        # Reading jailbreak queries
        jbs = []
        with open(jb_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = ast.literal_eval(line)
                    jbs.append(obj)

        for i, jb in enumerate(jbs):
            jb_query = jb['target_response_list'][0].replace('USER: <image> is a boat driving video. ', '').split(' GPT:<answer>')[0]
            
            if output != 'dinotxt':
                jb_text_embedding, jb_image_embedding = generate_embedding(image_path, jb_query, model, processor)
            else:
                jb_text_embedding, jb_image_embedding = generate_dinotxt_embedding(image_path, jb_query, dinotxt_model, dinotxt_tokenizer)
            jb_fused_embedding = fusion_model(jb_text_embedding, jb_image_embedding)
        
        np.savez(
            save_path,
            bq_fused=bq_fused_embedding.cpu().detach().numpy(),
            jb_fused=jb_fused_embedding.cpu().detach().numpy(),
            bq_text=bq_text_embedding.cpu().detach().numpy(),
            jb_text=jb_text_embedding.cpu().detach().numpy(),
            bq_image=bq_image_embedding.cpu().detach().numpy(),
            jb_image=jb_image_embedding.cpu().detach().numpy()
        )


def embedding_anymal(anymal_path, model, processor, output='embeddings'):
    # Initializing fusion model
    if output == 'embeddings':
        fusion_model = CrossAttentionFusion().to(device)
    elif output == 'siglip':
        fusion_model = CrossAttentionFusion(embed_dim=768, output_dim=768).to(device).half()
    elif output == 'blip':
        fusion_model = CrossAttentionFusion(embed_dim=512, output_dim=512).to(device)
    elif output == 'align':
        fusion_model = CrossAttentionFusion(embed_dim=640, output_dim=640).to(device)
    elif output == 'florence2' or output == 'openclip':
        fusion_model = CrossAttentionFusion(embed_dim=1024, output_dim=1024).to(device)
    elif output == 'dinotxt':
        fusion_model = CrossAttentionFusion(embed_dim=2048, output_dim=2048).to(device)
    fusion_model.load_state_dict(torch.load(f"./models/fusion_model_{output}.pth", map_location=device))
    fusion_model.eval()
    # Creating embedding path
    results_path = anymal_path.replace('datasets', output)
    os.makedirs(results_path, exist_ok=True)

    # Listing all videos
    mp4_files = []
    for root, dirs, files in os.walk(anymal_path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))

    for video_path in tqdm(sorted(mp4_files), desc='ANYMAL'):
        bq_path = video_path.replace('.mp4', '_bq.txt')
        jb_path = bq_path.replace('_bq.txt', '_jailbreak.json')

        filename = os.path.basename(bq_path)
        save_path = os.path.join(results_path, filename.replace('_bq.txt', '.npz'))

        if not os.path.exists(bq_path) or not os.path.exists(jb_path):
            continue

        # Reading benign queries
        with open(bq_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        bq_query = ""
        image_path = ""
        for line in lines:
            if line.startswith("QUERY: "):
                bq_query = line[len("QUERY: "):].strip()
            if line.startswith("IMAGES: "):
                image_path = line[len("IMAGES: "):].strip().split(',')[0]

        if output != 'dinotxt':
            bq_text_embedding, bq_image_embedding = generate_embedding(image_path, bq_query, model, processor)
        else:
            bq_text_embedding, bq_image_embedding = generate_dinotxt_embedding(image_path, bq_query, dinotxt_model, dinotxt_tokenizer)
        bq_fused_embedding = fusion_model(bq_text_embedding, bq_image_embedding)

        # Reading jailbreak queries
        jbs = []
        with open(jb_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = ast.literal_eval(line)
                    jbs.append(obj)

        for i, jb in enumerate(jbs):
            jb_query = jb['target_response_list'][0]
        
            if output != 'dinotxt':
                jb_text_embedding, jb_image_embedding = generate_embedding(image_path, jb_query, model, processor)
            else:
                jb_text_embedding, jb_image_embedding = generate_dinotxt_embedding(image_path, jb_query, dinotxt_model, dinotxt_tokenizer)
            jb_fused_embedding = fusion_model(jb_text_embedding, jb_image_embedding)
        
        np.savez(
            save_path,
            bq_fused=bq_fused_embedding.cpu().detach().numpy(),
            jb_fused=jb_fused_embedding.cpu().detach().numpy(),
            bq_text=bq_text_embedding.cpu().detach().numpy(),
            jb_text=jb_text_embedding.cpu().detach().numpy(),
            bq_image=bq_image_embedding.cpu().detach().numpy(),
            jb_image=jb_image_embedding.cpu().detach().numpy()
        )


def embedding_jb28k(jb28k_path, model, processor, output='embeddings'):
    # Loading dataset
    jb28k_df = pd.read_csv(jb28k_path + '/JailBreakV_28K.csv')
    # Initializing fusion model
    if output == 'embeddings':
        fusion_model = CrossAttentionFusion().to(device)
    elif output == 'siglip':
        fusion_model = CrossAttentionFusion(embed_dim=768, output_dim=768).to(device).half()
    elif output == 'blip':
        fusion_model = CrossAttentionFusion(embed_dim=512, output_dim=512).to(device)
    elif output == 'align':
        fusion_model = CrossAttentionFusion(embed_dim=640, output_dim=640).to(device)
    elif output == 'florence2' or output == 'openclip':
        fusion_model = CrossAttentionFusion(embed_dim=1024, output_dim=1024).to(device)
    elif output == 'dinotxt':
        fusion_model = CrossAttentionFusion(embed_dim=2048, output_dim=2048).to(device)
    fusion_model.load_state_dict(torch.load(f"./models/fusion_model_{output}.pth", map_location=device))
    fusion_model.eval()
    
    for index, row in tqdm(jb28k_df.iterrows(), total=len(jb28k_df), desc='JB28K'):
        # Extracting jailbreak type
        jb_type = row['from'].replace(' ', '_')
        results_path = f"{jb28k_path.replace('datasets', output)}/{jb_type}/{index}.npz"
        os.makedirs(os.path.dirname(f"{jb28k_path.replace('datasets', output)}/{jb_type}/"), exist_ok=True)

        if not os.path.exists(results_path):
            image_path = f'{jb28k_path}/{row["image_path"]}'

            if output != 'dinotxt':
                jb_text_embedding, jb_image_embedding = generate_embedding(image_path, row['jailbreak_query'], model, processor)
                rt_text_embedding, rt_image_embedding = generate_embedding(image_path, row['redteam_query'], model, processor)
            else:
                jb_text_embedding, jb_image_embedding = generate_dinotxt_embedding(image_path, row['jailbreak_query'], dinotxt_model, dinotxt_tokenizer)
                rt_text_embedding, rt_image_embedding = generate_dinotxt_embedding(image_path, row['redteam_query'], dinotxt_model, dinotxt_tokenizer)
            jb_fused_embedding = fusion_model(jb_text_embedding, jb_image_embedding)
            rt_fused_embedding = fusion_model(rt_text_embedding, rt_image_embedding)

            np.savez(
                results_path,
                jb_fused=jb_fused_embedding.cpu().detach().numpy(),
                rt_fused=rt_fused_embedding.cpu().detach().numpy(),
                jb_text=jb_text_embedding.cpu().detach().numpy(),
                rt_text=rt_text_embedding.cpu().detach().numpy(),
                jb_image=jb_image_embedding.cpu().detach().numpy(),
                rt_image=rt_image_embedding.cpu().detach().numpy()
            )


def embedding_lars(lars_path, model, processor, output='embeddings'):
    # Initializing fusion model
    if output == 'embeddings':
        fusion_model = CrossAttentionFusion().to(device)
    elif output == 'siglip':
        fusion_model = CrossAttentionFusion(embed_dim=768, output_dim=768).to(device).half()
    elif output == 'blip':
        fusion_model = CrossAttentionFusion(embed_dim=512, output_dim=512).to(device)
    elif output == 'align':
        fusion_model = CrossAttentionFusion(embed_dim=640, output_dim=640).to(device)
    elif output == 'florence2' or output == 'openclip':
        fusion_model = CrossAttentionFusion(embed_dim=1024, output_dim=1024).to(device)
    elif output == 'dinotxt':
        fusion_model = CrossAttentionFusion(embed_dim=2048, output_dim=2048).to(device)
    fusion_model.load_state_dict(torch.load(f"./models/fusion_model_{output}.pth", map_location=device))
    fusion_model.eval()
    # Creating embedding path
    results_path = f"{lars_path.replace('datasets', output)}"
    os.makedirs(results_path, exist_ok=True)

    # Listing all videos
    mp4_files = []
    for split in ['train', 'val', 'test']:
        lars_dir = os.path.join(lars_path, f'{split}/images_seq')
        for root, dirs, files in os.walk(lars_dir):
            for file in files:
                if file.endswith('.mp4'):
                    mp4_files.append(os.path.join(root, file))

    for video_path in tqdm(sorted(mp4_files), desc='LARS'):
        bq_path = video_path.replace('.mp4', '_bq.txt')
        jb_path = bq_path.replace('_bq.txt', '_jailbreak.json')
        
        filename = os.path.basename(bq_path)
        save_path = os.path.join(results_path, filename.replace('_bq.txt', '.npz'))
        
        if not os.path.exists(bq_path) or not os.path.exists(jb_path):
            continue
        
        # Reading benign queries
        with open(bq_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        bq_query = ""
        image_path = ""
        for line in lines:
            if line.startswith("QUERY: "):
                bq_query = line[len("QUERY: "):].strip()
            if line.startswith("IMAGES: "):
                image_path = line[len("IMAGES: "):].strip().split(',')[0]
        
        if output != 'dinotxt':
            bq_text_embedding, bq_image_embedding = generate_embedding(image_path, bq_query, model, processor)
        else:
            bq_text_embedding, bq_image_embedding = generate_dinotxt_embedding(image_path, bq_query, dinotxt_model, dinotxt_tokenizer)
        bq_fused_embedding = fusion_model(bq_text_embedding, bq_image_embedding)

        # Reading jailbreak queries
        jbs = []
        with open(jb_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = ast.literal_eval(line)
                    jbs.append(obj)

        for i, jb in enumerate(jbs):
            jb_query = jb['target_response_list'][0].replace('USER: <image> is a boat driving video. ', '').split(' GPT:<answer>')[0]
            
            if output != 'dinotxt':
                jb_text_embedding, jb_image_embedding = generate_embedding(image_path, jb_query, model, processor)
            else:
                jb_text_embedding, jb_image_embedding = generate_dinotxt_embedding(image_path, jb_query, dinotxt_model, dinotxt_tokenizer)
            jb_fused_embedding = fusion_model(jb_text_embedding, jb_image_embedding)

        np.savez(
            save_path,
            bq_fused=bq_fused_embedding.cpu().detach().numpy(),
            jb_fused=jb_fused_embedding.cpu().detach().numpy(),
            bq_text=bq_text_embedding.cpu().detach().numpy(),
            jb_text=jb_text_embedding.cpu().detach().numpy(),
            bq_image=bq_image_embedding.cpu().detach().numpy(),
            jb_image=jb_image_embedding.cpu().detach().numpy()
        )


def embedding_nuscenes(nuscenes_path, model, processor, output='embeddings'):
    # Initializing fusion model
    if output == 'embeddings':
        fusion_model = CrossAttentionFusion().to(device)
    elif output == 'siglip':
        fusion_model = CrossAttentionFusion(embed_dim=768, output_dim=768).to(device).half()
    elif output == 'blip':
        fusion_model = CrossAttentionFusion(embed_dim=512, output_dim=512).to(device)
    elif output == 'align':
        fusion_model = CrossAttentionFusion(embed_dim=640, output_dim=640).to(device)
    elif output == 'florence2' or output == 'openclip':
        fusion_model = CrossAttentionFusion(embed_dim=1024, output_dim=1024).to(device)
    elif output == 'dinotxt':
        fusion_model = CrossAttentionFusion(embed_dim=2048, output_dim=2048).to(device)
    fusion_model.load_state_dict(torch.load(f"./models/fusion_model_{output}.pth", map_location=device))
    fusion_model.eval()

    for folder in tqdm(sorted(os.listdir(nuscenes_path)), desc='ROBOPAIR'):
        folder_path = os.path.join(nuscenes_path, folder)
        results_path = f'{nuscenes_path.replace("datasets/nuscenes/videos", f"{output}/nuscenes")}'
        os.makedirs(results_path, exist_ok=True)
        if os.path.isdir(folder_path):
            json_path = os.path.join(folder_path, 'jailbreak.json')

            if os.path.exists(json_path):
                jbs = []
                with open(json_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            obj = ast.literal_eval(line)
                            jbs.append(obj)

                for i, jb in enumerate(jbs):
                    jb_query = jb['target_response_list'][0].replace('USER: <image> is a driving video. ', '').split(' GPT:<answer>')[0]
                    video_path = os.path.join(folder_path, 'video.mp4')
                    try:
                        image = extract_frame(video_path, frame_index=i)
                    except Exception as e:
                        continue

                    if output != 'dinotxt':
                        input = processor(text=jb_query, images=image, return_tensors="pt", padding=True, truncation=True).to(device)
                        outputs = model(**input)
                        text_embedding = outputs.text_embeds.to(device)
                        image_embedding = outputs.image_embeds.to(device)
                        fused_embedding = fusion_model(text_embedding, image_embedding)
                    else:
                        from dinov2.data.transforms import make_classification_eval_transform
                        image_preprocess = make_classification_eval_transform()
                        image_tensor = torch.stack([image_preprocess(image)], dim=0).cuda()

                        # Text
                        tokenized_texts_tensor = processor.tokenize(jb_query).cuda()

                        # Embedding
                        with torch.autocast('cuda', dtype=torch.float):
                            with torch.no_grad():
                                image_embedding = model.encode_image(image_tensor)
                                text_embedding = model.encode_text(tokenized_texts_tensor)
                        
                        fused_embedding = fusion_model(text_embedding, image_embedding)

                    np.savez(
                        os.path.join(results_path, f"{folder}_{i}.npz"),
                        jb_fused=fused_embedding.cpu().detach().numpy(),
                        jb_text=text_embedding.cpu().detach().numpy(),
                        jb_image=image_embedding.cpu().detach().numpy()
                    )


def embedding_vqa(vqa_path, model, processor, output='embeddings'):
    # Initializing fusion model
    if output == 'embeddings':
        fusion_model = CrossAttentionFusion().to(device)
    elif output == 'siglip':
        fusion_model = CrossAttentionFusion(embed_dim=768, output_dim=768).to(device).half()
    elif output == 'blip':
        fusion_model = CrossAttentionFusion(embed_dim=512, output_dim=512).to(device)
    elif output == 'align':
        fusion_model = CrossAttentionFusion(embed_dim=640, output_dim=640).to(device)
    elif output == 'florence2' or output == 'openclip':
        fusion_model = CrossAttentionFusion(embed_dim=1024, output_dim=1024).to(device)
    elif output == 'dinotxt':
        fusion_model = CrossAttentionFusion(embed_dim=2048, output_dim=2048).to(device)
    fusion_model.load_state_dict(torch.load(f"./models/fusion_model_{output}.pth", map_location=device))
    fusion_model.eval()
    # Reading dataset
    vqa_df = pd.read_csv(vqa_path + '/data.csv')

    for index, row in tqdm(vqa_df.iterrows(), total=len(vqa_df), desc='VQA'):
        results_path = f"{vqa_path.replace('datasets', output)}/{index}.npz"
        os.makedirs(f"{vqa_path.replace('datasets', output)}", exist_ok=True)

        if not os.path.exists(results_path):
            image_path = f'{vqa_path}/images/{row["image_id"]}.png'

            if output != 'dinotxt':
                bq_text_embedding, bq_image_embedding = generate_embedding(image_path, row['question'], model, processor)
            else:
                bq_text_embedding, bq_image_embedding = generate_dinotxt_embedding(image_path, row['question'], dinotxt_model, dinotxt_tokenizer)
            bq_fused_embedding = fusion_model(bq_text_embedding, bq_image_embedding)

            np.savez(
                results_path,
                bq_fused=bq_fused_embedding.cpu().detach().numpy(),
                bq_text=bq_text_embedding.cpu().detach().numpy(),
                bq_image=bq_image_embedding.cpu().detach().numpy()
            )


def embedding_wayve(wayve_path, model, processor, output='embeddings'):
    # Initializing fusion model
    if output == 'embeddings':
        fusion_model = CrossAttentionFusion().to(device)
    elif output == 'siglip':
        fusion_model = CrossAttentionFusion(embed_dim=768, output_dim=768).to(device).half()
    elif output == 'blip':
        fusion_model = CrossAttentionFusion(embed_dim=512, output_dim=512).to(device)
    elif output == 'align':
        fusion_model = CrossAttentionFusion(embed_dim=640, output_dim=640).to(device)
    elif output == 'florence2' or output == 'openclip':
        fusion_model = CrossAttentionFusion(embed_dim=1024, output_dim=1024).to(device)
    elif output == 'dinotxt':
        fusion_model = CrossAttentionFusion(embed_dim=2048, output_dim=2048).to(device)
    fusion_model.load_state_dict(torch.load(f"./models/fusion_model_{output}.pth", map_location=device))
    fusion_model.eval()
    # Reading dataset
    wayve_df = pd.read_parquet(wayve_path + '/train_redteam.parquet')

    for index, row in tqdm(wayve_df.iterrows(), total=len(wayve_df), desc='WAYVE'):
        results_path = f"{wayve_path.replace('datasets', output)}/{index}.npz"
        os.makedirs(f"{wayve_path.replace('datasets', output)}", exist_ok=True)

        if not os.path.exists(results_path):
            image_path = f'{wayve_path}/{row["images"][0]}'
            
            if output != 'dinotxt':
                bq_text_embedding, bq_image_embedding = generate_embedding(image_path, row['question'], model, processor)
            else:
                bq_text_embedding, bq_image_embedding = generate_dinotxt_embedding(image_path, row['question'], dinotxt_model, dinotxt_tokenizer)
            bq_fused_embedding = fusion_model(bq_text_embedding, bq_image_embedding)

            if output != 'dinotxt':
                rt_text_embedding, rt_image_embedding = generate_embedding(image_path, row['redteam'], model, processor)
            else:
                rt_text_embedding, rt_image_embedding = generate_dinotxt_embedding(image_path, row['redteam'], dinotxt_model, dinotxt_tokenizer)
            rt_fused_embedding = fusion_model(rt_text_embedding, rt_image_embedding)

            np.savez(
                results_path,
                bq_fused=bq_fused_embedding.cpu().detach().numpy(),
                rt_fused=rt_fused_embedding.cpu().detach().numpy(),
                bq_text=bq_text_embedding.cpu().detach().numpy(),
                rt_text=rt_text_embedding.cpu().detach().numpy(),
                bq_image=bq_image_embedding.cpu().detach().numpy(),
                rt_image=rt_image_embedding.cpu().detach().numpy()
            )


def run_all_embeddings(dataset_folder, model, processor, embed_dim, output_subfolder):
    jb28k_path = os.path.join(dataset_folder, "jb28k")
    vqa_path = os.path.join(dataset_folder, "vqa")

    model_save_path = f"./models/fusion_model_{output_subfolder}.pth"
    if not os.path.exists(model_save_path):
        train_cross_attention_model(jb28k_path, vqa_path, model, processor, model_save_path, embed_dim)

    embedding_aboships(os.path.join(dataset_folder, "aboships/images"), model, processor, output=output_subfolder)
    embedding_anymal(os.path.join(dataset_folder, "anymal"), model, processor, output=output_subfolder)
    embedding_jb28k(jb28k_path, model, processor, output=output_subfolder)
    embedding_lars(os.path.join(dataset_folder, "lars"), model, processor, output=output_subfolder)
    embedding_nuscenes(os.path.join(dataset_folder, "nuscenes/videos"), model, processor, output=output_subfolder)
    embedding_vqa(vqa_path, model, processor, output=output_subfolder)
    embedding_wayve(os.path.join(dataset_folder, "wayve"), model, processor, output=output_subfolder)

if __name__ == "__main__":
    setSeed()

    # --- Run CLIP version ---
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    run_all_embeddings(dataset_folder, clip_model, clip_processor, embed_dim=512, output_subfolder="embeddings")

    # --- Run SigLIP version ---
    siglip_model = AutoModel.from_pretrained(
        "google/siglip-base-patch16-224",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    run_all_embeddings(dataset_folder, siglip_model, siglip_processor, embed_dim=768, output_subfolder="siglip")

    # --- Run BLIP version ---
    blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    run_all_embeddings(dataset_folder, blip_model, blip_processor, embed_dim=512, output_subfolder="blip")

    # --- Run ALIGN (if available on HF) ---
    align_model = AutoModel.from_pretrained("kakaobrain/align-base").to(device)
    align_processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
    run_all_embeddings(dataset_folder, align_model, align_processor, embed_dim=640, output_subfolder="align")

    # --- Run OpenCLIP large model ---
    openclip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
    openclip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    run_all_embeddings(dataset_folder, openclip_model, openclip_processor, embed_dim=1024, output_subfolder="openclip")

    # --- Run DinoTXT model ---
    dinotxt = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l').to(device)
    from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer
    dinotxt_model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().cuda()
    dinotxt_tokenizer = get_tokenizer()
    run_all_embeddings(dataset_folder, dinotxt_model, dinotxt_tokenizer, embed_dim=2048, output_subfolder="dinotxt")
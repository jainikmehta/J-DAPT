import torch
import torch.nn as nn
from torch.utils.data import Dataset



class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, output_dim=512):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 3, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, text_embeds, image_embeds):
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(1)
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(1)

        attended_text, _ = self.cross_attn(text_embeds, image_embeds, image_embeds)  # (B, 1, D)

        fused = torch.cat([text_embeds, image_embeds, attended_text], dim=-1)  # (B, 1, 3D)
        fused = self.projection(fused)  # (B, 1, output_dim)
        
        return fused.squeeze(1)  # (B, output_dim)


class MultimodalEmbeddingDataset(Dataset):
    def __init__(self, image_embeds, text_embeds):
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds

    def __len__(self):
        return len(self.image_embeds)

    def __getitem__(self, idx):
        return self.text_embeds[idx], self.image_embeds[idx]
    

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

class DeepNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
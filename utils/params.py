import os
import torch



# Device if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
seed = 151836

# Server LLM / VLM Config (risl lab server)
llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
llm_api_key_path = os.path.expanduser("~/.config/llm/api_key")
vlm_model = os.getenv("VLM_MODEL", "gemma-4-31b")
llm_model = os.getenv("LLM_MODEL", "qwen3-vl-32b")

# Folders
dataset_folder = "./datasets"
# embeddings_folder = dataset_folder.replace("datasets", "embeddings")

# Dataset URLs
aboships_link = 'https://zenodo.org/records/8383205/files/ABOships-PLUS.zip?download=1'
jb28k_link = 'PLACEHOLDER_JB28K_LINK'  # Replace with the actual link after access request
lars_link = 'https://box.vicos.si/lars/lars_v1.0.0_images_seq.zip'
nuscenes_link = 'https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz'
vqa_link = 'tezansahu/processed-daquar-dataset'
wayve_link = 'https://drive.google.com/drive/folders/1GiwWGfrM8pO27CYLu_9Uwtdcz0JoqHr7'
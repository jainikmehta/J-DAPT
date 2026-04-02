import torch



# Device if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
seed = 151836

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
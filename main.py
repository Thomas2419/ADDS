import os
import torch
from data import extract_labels, JSONLabelDataset
from model import ADDSModel, AsymmetricLoss
from training_utils import load_model, train_model
import open_clip
import json
from torch.utils.data import random_split, DataLoader

def generate_text_embeddings(labels, clip_model, tokenizer, device):
    # Create text prompts and encode them
    text_prompts = [f"This photo contains {label}" for label in labels]
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(tokenizer(text_prompts).to(device))

    return text_embeddings


def main():
    # Configuration settings
    data_dir = "" # path for dataloader
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = None
    num_epochs = 2000
    tags_path = ""  # Path for tags.json

    # Load tags
    with open(tags_path, 'r') as f:
        comprehensive_data = json.load(f)
    
    possible_labels = extract_labels(comprehensive_data)
    num_labels = len(possible_labels)
    preprocess_json_files(data_dir, possible_labels)

    # Initialize the OpenCLIP model once
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    clip_model = clip_model.to(device)
    clip_model.eval()  # Set to evaluation mode as it's used for encoding
    tokenizer = open_clip.get_tokenizer(model_name)

    # Generate text embeddings for all labels
    text_embeddings = generate_text_embeddings(possible_labels, clip_model, tokenizer, device)

    # Wrap the OpenCLIP model with your custom ADDSModel
    if model_path and os.path.exists(model_path):
        print("Starting training from " + model_path)
        model = load_model(model_path, device, clip_model, embed_dim=768, num_heads=16, num_layers=16, num_labels=num_labels)
    else:
        print("Starting training from scratch.")
        model = ADDSModel(clip_model=clip_model, embed_dim=768, num_heads=32, num_layers=6, num_labels=num_labels).to(device)

    # Pass the correct model instance to the dataset
    dataset = JSONLabelDataset(data_dir=data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4)

    # Optimizer setup
    optimizer = torch.optim.AdamW([
        {'params': model.base_model.visual.parameters(), 'lr': 5e-8, 'weight_decay': 1e-2},
        {'params': model.decoder.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2},
        {'params': model.final_fc.parameters(), 'lr': 1e-6, 'weight_decay': 1e-2},
    ])

    # Loss function
    criterion = AsymmetricLoss()

    # Start training and pass the text_embeddings
    train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=num_epochs, checkpoint_dir="checkpoints", text_embeddings=text_embeddings)

if __name__ == "__main__":
    main()


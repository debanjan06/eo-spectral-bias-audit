import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from dataset import AgriSightDataset
from models.multi_modal_cnn import MultiModalCNN

def train_model():
    print("LOG: Initializing training pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LOG: Training hardware: {device}")

    # Configuration
    batch_size = 32
    learning_rate = 0.001
    epochs = 10
    
    # Data Loading Strategy
    # Using the standardized naming convention established in src/dataset.py
    metadata_df = pd.read_csv('data/processed/agrisight_training_dataset.csv')
    patches_dir = 'data/processed/california_patches'
    
    dataset = AgriSightDataset(metadata_df=metadata_df, patches_dir=patches_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MultiModalCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            # Correctly unpacking the dictionary returned by AgriSightDataset
            spatial = batch['spatial'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through the multi-modal architecture
            outputs = model(spatial, tabular)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"PROGRESS: Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Model Persistence
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/best_baseline_model.pth')
    print("SUCCESS: Training completed. Weights saved to models/best_baseline_model.pth")

if __name__ == "__main__":
    train_model()

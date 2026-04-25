import os
import torch
import torch.nn as nn
import torch.optim as optim
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
    
    # Data loading
    dataset = AgriSightDataset(csv_file='data/processed/agriguard_training_dataset.csv')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MultiModalCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (spatial, tabular, labels) in enumerate(train_loader):
            spatial, tabular, labels = spatial.to(device), tabular.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spatial, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"PROGRESS: Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Weights preservation
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/best_baseline_model.pth')
    print("SUCCESS: Training completed. Weights exported to models/best_baseline_model.pth")

if __name__ == "__main__":
    train_model()

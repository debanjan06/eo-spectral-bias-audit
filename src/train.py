import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom dataset engine and architecture
from evaluate_baseline import CropHealthDataset 
from models.multi_modal_cnn import MultiModalCNN

def train_baseline():
    print("🧠 Initiating Model Training on California Baseline...")
    os.makedirs('models', exist_ok=True)
    
    # 1. Setup Model, Optimizer, and Loss Function
    model = MultiModalCNN(num_classes=3)
    
    # Auto-detect GPU for massive speedup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 2. Load the REAL Training Data
    print("📊 Connecting to AgriGuard dataset...")
    try:
        dataset = CropHealthDataset(
            folder_path='data/processed/california_patches', 
            csv_path='data/processed/agriguard_training_dataset.csv' # Pointing to the data folder!
        )
    except FileNotFoundError:
        print("❌ Error: Could not find 'data/agriguard_training_dataset.csv'")
        print("Please ensure the CSV was moved into the 'data' folder, and that you are running this script from the root eo-spectral-bias-audit folder.")
        return

    # Group data into batches of 32 to prevent memory crashes
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    epochs = 5 # 5 full passes over the dataset
    
    # 3. The Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for spatial_batch, tabular_batch, labels in train_loader:
            # Move data to the GPU
            spatial_batch = spatial_batch.to(device)
            tabular_batch = tabular_batch.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad() # Clear old gradients
            
            # Forward pass: the model makes its guesses
            outputs = model(spatial_batch, tabular_batch)
            loss = criterion(outputs, labels) # Calculate how wrong the guesses were
            
            # Backward pass: adjust the weights to learn
            loss.backward()
            optimizer.step()
            
            # Track real-time accuracy for this batch
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # Print metrics at the end of each epoch
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}%")

    # 4. Save the trained weights to your hard drive
    torch.save(model.state_dict(), 'models/best_baseline_model.pth')
    print("\n✅ Training Complete. Model weights saved to models/best_baseline_model.pth")

if __name__ == "__main__":
    train_baseline()
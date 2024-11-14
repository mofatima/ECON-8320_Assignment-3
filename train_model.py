
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Define custom dataset class
class CustomFashionDataset(Dataset):
    def __init__(self, is_train=True, transform=None):
        self.data = datasets.FashionMNIST(root='./data', train=is_train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lbl = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, lbl

# Define data transformations
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create custom data loaders
train_loader = DataLoader(CustomFashionDataset(is_train=True, transform=image_transform), batch_size=64, shuffle=True)
test_loader = DataLoader(CustomFashionDataset(is_train=False, transform=image_transform), batch_size=64, shuffle=False)

# Define the neural network with Batch Normalization and Dropout
class CustomFashionNet(nn.Module):
    def __init__(self):
        super(CustomFashionNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(64, 10)
   
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.drop1(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.drop2(x)
        x = torch.relu(self.bn3(self.layer3(x)))
        x = self.drop3(x)
        x = self.output_layer(x)
        return x

# Initialize model, loss function, and optimizer
fashion_model = CustomFashionNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fashion_model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop with increased epochs
num_epochs = 10
for epoch in range(num_epochs):
    fashion_model.train()  # Set model to training mode
    running_loss = 0.0
    for imgs, lbls in train_loader:
        optimizer.zero_grad()
        outputs = fashion_model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Step the scheduler after each epoch
    scheduler.step()
    print(f"Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}")

# Save model weights
torch.save(fashion_model.state_dict(), 'fashion_model_weights.pth')



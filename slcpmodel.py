# Import dependencies
import os
# Format data
import pandas as pd
# Create Model Architecture 
import torch
from torch import nn, save, load
from torch.optim import Adam
# Load Dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
# Transform the Images for Training
from torchvision.transforms import ToTensor, Compose, Resize
# Access and Process Images
from PIL import Image
from skimage import io

# Prepare the custom dataset
class LungCancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Turn the csv file into a DataFrame
        self.annotations = pd.read_csv(csv_file)

        # Other Settings
        self.root_dir = root_dir
        self.transform = transform

    # Finds the length of the dataset
    def __len__(self):
        return len(self.annotations)
        
    # When an image is being accessed
    def __getitem__(self, index):
        # Get Image, and match it with it's Y label
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        # Apply Transformations
        if self.transform:
            image = self.transform(image)

        # Return the image and y label
        return (image, y_label)
    
# Load Data
dataset = LungCancerDataset(csv_file='lcdataset.csv', root_dir='data', transform = transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [12500, 2499])
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)
    
# Create Neural Network Architecture
class LCModel(nn.Module):
    def __init__(self):
        # Image size is 768 x 768
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Flatten(),
            
            nn.Linear(64*189*189, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        return self.model(x)
    

# Create instances of the opimizer, loss, and neural network
lcm = LCModel()
lcm.to('cpu')
opt = Adam(lcm.parameters(), lr=1e-3)
loss_func = nn.CrossEntropyLoss()

# Training Loop
if __name__ == '__main__':
    for epoch in range(10): # Train for 10 epochs
        for batch in train_loader:
            # Get batch
            X,y = batch
            X,y = X.to('cpu'), y.to('cpu')

            # Feed the data to the model and calculate loss
            yhat = lcm(X)
            loss = loss_func(yhat, y)
            
            # Apply backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # Print when each epoch is complete    
        print(f"Epoch:{epoch} has a loss of {loss.item()}")
        
# Save the model         
with open('lcpmodel.pt', 'wb') as f:
    save(lcm.state_dict(), f)

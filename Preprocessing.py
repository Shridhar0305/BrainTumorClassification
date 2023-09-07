# The BraTS 2019 dataset is changed from a 16 bit signed integers to 32 bit floating point numbers. The dataset is preprocessed to care for factors such as patients positins in the scanner,
# and other issues that can cause differences in brightness on the MRI. These are cared for by employing SimpleITK N4 bias correction filter on all images. 

# Next steps will include: patch extraction and kernel methods for SVM training for K fold training/80 % training-20% testing. This will then be further evaluated using evaluations. 
# next steps include getting the feature vector and the labels and training the svm using kernel. 

# Now the next steps is to train the model and get the feautures after training it for the svm.

import SimpleITK as sitk
import numpy as np
import os
from skimage.util import view_as_windows
import random
from scipy.ndimage import rotate
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def apply_n4_correction(input_path, output_path):
    input_image = sitk.ReadImage(input_path)
    input_image_float = sitk.Cast(input_image, sitk.sitkFloat32)  # Convert to 32-bit float
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(input_image_float)
    sitk.WriteImage(corrected_image, output_path)

def extract_patches(image, patch_size, stride):
    image_array = sitk.GetArrayFromImage(image)  # Convert SimpleITK image to NumPy array
    patches = view_as_windows(image_array, patch_size, step=stride)
    return patches

    
input_directory = "/Downloads/BrainTumorSegmentation/InputImages"
output_directory = "/Downloads/BrainTumorSegmentation/OutputImages"
patch_size = (64, 64, 64)
stride = (32, 32, 32)
rotation_angles = [0, 90, 180, 270]

for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith(".nii"):  # Adjust file extension if needed
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_directory)
            output_path = os.path.join(output_directory, relative_path)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            apply_n4_correction(input_path, output_path)
            corrected_image = sitk.ReadImage(output_path)
            patches = extract_patches(corrected_image, patch_size, stride)

            for i, patch in enumerate(patches):
                # Augment the data with rotated patches
                for angle in rotation_angles:
                    rotated_patch = rotate(patch, angle, reshape=False)

                    # Append the original and rotated patches as pairs
                    data_points.append(patch)
                    labels.append(angle)

                    data_points.append(rotated_patch)
                    labels.append(angle)

print("Data points and corresponding labels created.")



class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data_points  # List of data points (e.g., patches or images)
        self.labels = labels  # List of corresponding labels (e.g., rotation angles)
        self.transform = transform  # Optional data transformations (e.g., resizing, normalization)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Apply transformations (if provided)
        if self.transform:
            sample = self.transform(sample)

        return sample, labeld

# Define the neural network architecture for rotation prediction
class RotationPredictionNet(nn.Module):
    def __init__(self, num_rotation_angles):
        super(RotationPredictionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_rotation_angles)  # Predict rotation angles

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data preparation
# Assuming you have a dataset containing pairs of original and rotated patches
# You'll need to load this dataset and create labels indicating the rotation angles

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create DataLoader objects for training and validation
batch_size = 32
train_dataset = CustomDataset(X_train, y_train)  # Replace with your dataset class
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomDataset(X_val, y_val)  # Replace with your dataset class
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the rotation prediction model
num_rotation_angles = 4  # 0, 90, 180, 270 degrees
rotation_model = RotationPredictionNet(num_rotation_angles)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rotation_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    rotation_model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = rotation_model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate average loss for the epoch
    average_loss = total_loss / len(train_loader)

    # Validation
    rotation_model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in val_loader:
            outputs = rotation_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        average_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{num_epochs}] | Training Loss: {average_loss:.4f} | Validation Loss: {average_val_loss:.4f}")

# Feature extraction
# Now, you can extract features from the intermediate layers of the rotation_model
# These features can be used as input for your SVM model or other downstream tasks

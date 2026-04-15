**pgm 1**
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. GENERATE SYNTHETIC DATA
# Create 100 points between 0 and 10
X = torch.linspace(0, 10, 100).reshape(-1, 1) 
# Calculate y using true slope 2.5 and bias 5.0, then add some random noise [cite: 58, 59, 62]
y = 2.5 * X + 5.0 + torch.randn(100, 1) 

# 2. DEFINE THE MODEL
# nn.Linear(1, 1) automatically creates W and b for 1 input and 1 output [cite: 12, 13]
model = nn.Linear(in_features=1, out_features=1)

# 3. LOSS FUNCTION AND OPTIMIZER
# MSELoss measures the Mean Squared Error 
criterion = nn.MSELoss() 
# SGD (Stochastic Gradient Descent) updates W and b using the learning rate 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. TRAINING LOOP
epochs = 100
for epoch in range(epochs):
    # Predict: Calculate Y_hat = XW + b [cite: 15, 45]
    y_pred = model(X)
    
    # Compute Loss: Difference between predicted and actual y [cite: 47, 92]
    loss = criterion(y_pred, y)
    
    # Backpropagation (The Gradient Descent steps)
    optimizer.zero_grad()  # Clear old gradients from previous step
    loss.backward()        # Compute gradients (dLoss/dW and dLoss/db) [cite: 20, 21]
    optimizer.step()       # Update W and b using the gradients [cite: 24, 49]

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. VISUALIZATION [cite: 30, 31]
plt.scatter(X, y, label="Original Data")
plt.plot(X, model(X).detach(), color='red', label="Fitted Line")
plt.legend()
plt.show()

# Print the final learned values [cite: 115]
print(f"Final W: {model.weight.item():.2f}, Final b: {model.bias.item():.2f}")


**pgm 2**
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. LOAD AND PREPROCESS DATA
iris = load_iris()
X, y = iris.data, iris.target

# Standardize features (Mean=0, Variance=1) [cite: 181, 241]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into Training and Testing sets (80% train, 20% test) [cite: 245, 246]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert NumPy arrays to PyTorch Tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train) # LongTensor is required for CrossEntropy
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# 2. DEFINE MODEL
# Softmax Regression is just a Linear layer with CrossEntropyLoss [cite: 176, 212]
# Input: 4 features, Output: 3 classes
model = nn.Linear(4, 3) 

# 3. LOSS AND OPTIMIZER
# CrossEntropyLoss in PyTorch automatically applies Softmax! [cite: 183, 185, 217]
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. TRAINING LOOP [cite: 194, 214]
for epoch in range(100):
    # Forward Pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward Pass (The Trio)
    optimizer.zero_grad() # Clear gradients [cite: 198]
    loss.backward()      # Compute gradients [cite: 197, 218]
    optimizer.step()     # Update weights and bias [cite: 199, 219]

# 5. EVALUATE ACCURACY [cite: 186, 261]
with torch.no_grad():
    test_outputs = model(X_test)
    # Get class with highest probability [cite: 228, 287]
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")



  **pgm 3**
  import torch
import torch.nn as nn

# 1. DEFINE XOR DATASET
# Inputs: 4 possible combinations of 0 and 1
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32) 
# Outputs: 0 if inputs are same, 1 if inputs are different
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 2. DEFINE THE MLP MODEL
# We need a hidden layer to handle the non-linear XOR problem
class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input(2) -> Hidden(2)
        self.hidden = nn.Linear(2, 2) 
        # Non-linear activation (Sigmoid)
        self.sigmoid = nn.Sigmoid()   
        # Hidden(2) -> Output(1)
        self.output = nn.Linear(2, 1) 

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = XORModel()

# 3. LOSS AND OPTIMIZER
# MSELoss is used for binary output prediction in this lab
criterion = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. TRAINING LOOP
for epoch in range(5000): # XOR often needs more epochs to converge
    # Forward Pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward Pass (The Trio)
    optimizer.zero_grad() 
    loss.backward()      
    optimizer.step()     

# 5. PREDICTION
with torch.no_grad():
    results = model(X)
    # Round to nearest integer (0 or 1) to get final logic output
    print("Final Predictions:\n", torch.round(results))

**pgm 4**
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 1. SETUP PARAMETERS (Based on Lab Manual)
# input_size 416 comes from 13 MFCC coefficients * 32 time-steps [cite: 1887, 1897]
input_size = 416 
num_classes = 5
class_names = ["yes", "no", "up", "down", "left"] 
# --- Note for Exam: In a real lab, you would load your MFCC tensors here ---
# Creating dummy data so the code is runnable for your practice
X_train = torch.randn(100, input_size) 
y_train = torch.randint(0, num_classes, (100,))
X_test = torch.randn(20, input_size)
y_test = torch.randint(0, num_classes, (20,))

# 2. DEFINE THE DNN ARCHITECTURE [cite: 1888, 1889]
class SpeechDNN(nn.Module):
    def __init__(self):
        super(SpeechDNN, self).__init__()
        self.network = nn.Sequential(
            # Layer 1: Dense(512) + ReLU + Dropout [cite: 1888]
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: Dense(256) + ReLU + Dropout [cite: 1888]
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3: Dense(128) + ReLU [cite: 1889]
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Output Layer [cite: 1889]
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = SpeechDNN()

# 3. LOSS AND OPTIMIZER [cite: 1890]
criterion = nn.CrossEntropyLoss() # Automatically handles Softmax
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. TRAINING LOOP (20 Epochs) [cite: 1890, 1901]
print("Starting Training...")
for epoch in range(20):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass (The Exam Trio)
    optimizer.zero_grad() # 1. Clear gradients
    loss.backward()      # 2. Compute gradients
    optimizer.step()     # 3. Update weights
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/20], Loss: {loss.item():.4f}")

# 5. EVALUATION: METRICS [cite: 1891, 1893, 1901]
print("\nEvaluating Model...")
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    test_outputs = model(X_test)
    # Pick class with highest probability score [cite: 1851, 1896, 1905]
    _, predicted = torch.max(test_outputs, 1)
    
    y_true.extend(y_test.numpy())
    y_pred.extend(predicted.numpy())

# Final Results
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred)) 



**pgm 5a**
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. DATA PREPROCESSING
# Convert images to tensors and normalize pixel values to 0-1 range
transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST Training and Test datasets
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoaders for batch processing
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

# 2. DEFINE CNN MODEL
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1: Convolution + ReLU + MaxPool
            nn.Conv2d(1, 32, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Layer 2: Convolution + ReLU + MaxPool
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Layer 3: Flatten + Dense layers
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 10) # 10 output classes for digits 0-9
        )

    def forward(self, x):
        return self.network(x)

model = SimpleCNN()

# 3. LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. TRAINING LOOP
print("Training started...")
for epoch in range(2): # Shortened for exam; use 5 for better accuracy
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad() # The "Trio": 1. Clear gradients
        loss.backward()      # 2. Compute gradients
        optimizer.step()     # 3. Update weights
    print(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")

# 5. EVALUATION: CLASSIFICATION REPORT & CONFUSION MATRIX
print("\nEvaluating model...")
model.eval() # Set to evaluation mode
y_true = []
y_pred = []

with torch.no_grad(): # Disable gradient calculation for testing
    for images, labels in test_loader:
        outputs = model(images)
        # Use argmax to get the predicted digit class
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Display Final Metrics
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred))



**pgm 5b**
import numpy as np

# Basic 5x5 Image and 3x3 Kernel
image = np.array([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]])
kernel = np.array([[1,0,1],[0,1,0],[1,0,1]])

# Calculate Output Dimensions (Valid: P=0, S=1)
# (5 - 3) / 1 + 1 = 3
output = np.zeros((3, 3))

# SLIDING WINDOW (Manual Step 6-10)
for i in range(3):     # Vertical slide
    for j in range(3): # Horizontal slide
        # Extract a "patch" (region) of the image
        patch = image[i:i+3, j:j+3]
        # Multiply element-wise and Sum (The Dot Product)
        output[i, j] = np.sum(patch * kernel)

print("Output Feature Map:\n", output)


**pgm 6**
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. PREPROCESSING & DATA LOADING
# We resize to 75x75 as required by Inception V3
transform = transforms.Compose([
    transforms.Resize((299, 299)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load Training and Test sets (CIFAR-10 has 10 classes)
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

class_names = train_set.classes # ['airplane', 'automobile', 'bird', ...]

# 2. LOAD PRE-TRAINED GOOGLENET (Inception V3)
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)

# 3. FREEZE BASE LAYERS (Transfer Learning)
for param in model.parameters():
    param.requires_grad = False

# 4. REPLACE THE TOP LAYER
# Change final layer 'fc' to match CIFAR-10 (10 classes)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)

# 5. LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 6. TRAINING LOOP
print("Starting Training...")
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        outputs, aux_outputs = model(images)
        loss_main = criterion(outputs, labels)
        loss_aux = criterion(aux_outputs, labels)
        loss = loss_main + 0.4 * loss_aux

        optimizer.zero_grad() # The "Trio"
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")

# 7. EVALUATION: METRICS
print("\nGenerating Metrics...")
model.eval() # Disable dropout for evaluation
y_true = []
y_pred = []

with torch.no_grad(): # Disable gradients to save memory
    for images, labels in test_loader:
        outputs = model(images)
        # Use argmax to find the predicted class
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# 8. PRINT PERFORMANCE REPORTS
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred))

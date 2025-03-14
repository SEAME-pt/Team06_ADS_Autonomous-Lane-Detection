# PyTorch Fundamentals

## Introduction to PyTorch
- **What is PyTorch?**
  - PyTorch is an open-source machine learning library developed by Facebook's AI Research lab (FAIR).
  - It is widely used for deep learning, scientific computing, and tensor computations.
  - Key features:
    - Dynamic computation graphs (define-by-run).
    - Strong GPU acceleration support.
    - Integration with Python libraries like NumPy.

- **Why PyTorch?**
  - Easy to learn and use, especially for Python developers.
  - Excellent for research and prototyping due to its flexibility.
  - Strong community and ecosystem (e.g., torchvision, torchaudio).

---

## Tensors: The Building Blocks of PyTorch
- **What are Tensors?**
  - Tensors are multi-dimensional arrays, similar to NumPy arrays, but with GPU support.
  - They are the primary data structure in PyTorch.

- **Creating Tensors**
  ```python
  import torch

  # From a list
  tensor = torch.tensor([1, 2, 3])

  # Empty tensor (uninitialized)
  empty_tensor = torch.empty(2, 3)

  # Tensor of zeros
  zeros = torch.zeros(2, 3)

  # Tensor of ones
  ones = torch.ones(2, 3)

  # Random tensor (uniform distribution)
  rand_tensor = torch.rand(2, 3)

  # Random tensor (normal distribution)
  randn_tensor = torch.randn(2, 3)

  # Tensor with a range of values
  range_tensor = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
  ```

- **Tensor Attributes**
  - Shape: `tensor.shape`
  - Data type: `tensor.dtype`
  - Device: `tensor.device` (CPU or GPU).

- **Tensor Operations**
  - **Element-wise operations**:
    ```python
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = a + b  # Element-wise addition
    ```
  - **Matrix multiplication**:
    ```python
    mat1 = torch.tensor([[1, 2], [3, 4]])
    mat2 = torch.tensor([[5, 6], [7, 8]])
    result = torch.matmul(mat1, mat2)  # Or use @ operator: mat1 @ mat2
    ```
  - **Reshaping**:
    ```python
    tensor = torch.arange(0, 9)
    reshaped = tensor.view(3, 3)  # Reshape to 3x3
    flattened = tensor.flatten()   # Flatten to 1D
    ```

- **GPU Support**
  - Move tensors to GPU:
    ```python
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = tensor.to(device)
    ```

---

## Autograd: Automatic Differentiation
- **What is Autograd?**
  - PyTorch’s automatic differentiation engine for computing gradients.
  - It tracks operations on tensors with `requires_grad=True` and computes gradients during the backward pass.

- **How Autograd Works**
  - Enable gradient tracking:
    ```python
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    ```
  - Perform operations:
    ```python
    y = x.sum()  # y = x[0] + x[1] + x[2]
    ```
  - Compute gradients:
    ```python
    y.backward()  # Computes gradients of y w.r.t. x
    print(x.grad)  # Output: tensor([1., 1., 1.])
    ```

- **Disabling Gradient Tracking**
  - Use `torch.no_grad()` for inference or evaluation:
    ```python
    with torch.no_grad():
        y = x * 2  # No gradient tracking
    ```

---

## Neural Network Module (`torch.nn`)
- **What is `torch.nn`?**
  - PyTorch’s module for building neural networks.
  - Provides pre-defined layers, loss functions, and optimizers.

- **Defining a Neural Network**
  - Subclass `nn.Module`:
    ```python
    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.layer1 = nn.Linear(10, 50)  # Fully connected layer
            self.layer2 = nn.Linear(50, 1)   # Output layer

        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)  # Activation function
            x = self.layer2(x)
            return x
    ```

- **Loss Functions**
  - Mean Squared Error (MSE): `nn.MSELoss()`
  - Binary Cross-Entropy: `nn.BCELoss()`
  - Cross-Entropy: `nn.CrossEntropyLoss()`

- **Optimizers**
  - Stochastic Gradient Descent (SGD): `torch.optim.SGD(model.parameters(), lr=0.01)`
  - Adam: `torch.optim.Adam(model.parameters(), lr=0.001)`

---

# PyTorch Workflow

## Data Preparation
- **Datasets and DataLoaders**
  - Use `torch.utils.data.Dataset` to create custom datasets.
  - Use `torch.utils.data.DataLoader` to load data in batches.
  ```python
  from torch.utils.data import DataLoader, TensorDataset

  # Create a dataset
  dataset = TensorDataset(features, labels)

  # Create a DataLoader
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
  ```

## Training Loop
1. **Forward Pass**: Compute predictions.
2. **Compute Loss**: Compare predictions with ground truth.
3. **Backward Pass**: Compute gradients using `.backward()`.
4. **Update Weights**: Use an optimizer to update model parameters.
   ```python
   model = MyModel()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.MSELoss()

   for epoch in range(epochs):
       for batch_features, batch_labels in dataloader:
           optimizer.zero_grad()  # Clear gradients
           outputs = model(batch_features)  # Forward pass
           loss = criterion(outputs, batch_labels)  # Compute loss
           loss.backward()  # Backward pass
           optimizer.step()  # Update weights
   ```

## Evaluation
- Test the model on unseen data:
  ```python
  model.eval()  # Set model to evaluation mode
  with torch.no_grad():
      test_outputs = model(test_features)
      test_loss = criterion(test_outputs, test_labels)
  ```

## Saving and Loading Models
- **Save Model**:
  ```python
  torch.save(model.state_dict(), "model.pth")
  ```
- **Load Model**:
  ```python
  model = MyModel()
  model.load_state_dict(torch.load("model.pth"))
  ```

---

# PyTorch Classification

## Binary Classification
- **Output Layer**: Single neuron with sigmoid activation.
  ```python
  self.output = nn.Linear(hidden_units, 1)
  ```
- **Loss Function**: Binary Cross-Entropy Loss (`nn.BCELoss`).

## Multi-Class Classification
- **Output Layer**: Softmax activation.
  ```python
  self.output = nn.Linear(hidden_units, num_classes)
  ```
- **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`).

## Metrics
- **Accuracy**: Percentage of correct predictions.
- **Confusion Matrix**: For detailed performance analysis.

---

# PyTorch Computer Vision

## Convolutional Neural Networks (CNNs)
- **Layers**:
  - Convolutional: `nn.Conv2d(in_channels, out_channels, kernel_size)`
  - Pooling: `nn.MaxPool2d(kernel_size)`
  - Flatten: `nn.Flatten()`
  - Fully Connected: `nn.Linear(in_features, out_features)`

## Datasets and Transformations
- **Datasets**:
  ```python
  from torchvision import datasets

  train_data = datasets.MNIST(root="data", train=True, download=True)
  ```
- **Transformations**:
  ```python
  from torchvision import transforms

  transform = transforms.Compose([
      transforms.Resize((28, 28)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])
  ```

## Transfer Learning
- Use pre-trained models:
  ```python
  from torchvision import models

  model = models.resnet18(pretrained=True)
  model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
  ```

---

## PyTorch Custom Datasets

### Creating Custom Datasets
- Subclass `torch.utils.data.Dataset`:
  ```python
  from torch.utils.data import Dataset

  class CustomDataset(Dataset):
      def __init__(self, data, labels, transform=None):
          self.data = data
          self.labels = labels
          self.transform = transform

      def __len__(self):
          return len(self.data)

      def __getitem__(self, idx):
          sample = self.data[idx]
          label = self.labels[idx]
          if self.transform:
              sample = self.transform(sample)
          return sample, label
  ```

### Using Custom Datasets
- Create a dataset and DataLoader:
  ```python
  dataset = CustomDataset(data, labels, transform=transform)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
  ```

---

## PyTorch Going Modular

### Organizing Code into Modules
- Split code into reusable modules:
  - `model.py`: Define the model architecture.
  - `train.py`: Training loop and evaluation.
  - `utils.py`: Utility functions (e.g., data loading, transformations).

### Example: Modular Workflow
1. **Model Definition** (`model.py`):
   ```python
   import torch.nn as nn

   class MyModel(nn.Module):
       def __init__(self):
           super(MyModel, self).__init__()
           self.layer1 = nn.Linear(10, 50)
           self.layer2 = nn.Linear(50, 1)

       def forward(self, x):
           x = self.layer1(x)
           x = torch.relu(x)
           x = self.layer2(x)
           return x
   ```

2. **Training Script** (`train.py`):
   ```python
   from model import MyModel
   from torch.utils.data import DataLoader
   import torch.optim as optim
   import torch.nn as nn

   # Load data
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # Initialize model, optimizer, and loss function
   model = MyModel()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.MSELoss()

   # Training loop
   for epoch in range(epochs):
       for batch_features, batch_labels in dataloader:
           optimizer.zero_grad()
           outputs = model(batch_features)
           loss = criterion(outputs, batch_labels)
           loss.backward()
           optimizer.step()
   ```

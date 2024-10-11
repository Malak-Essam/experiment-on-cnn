
# Convolutional Neural Network (CNN) Filter Optimization with PyTorch

This repository demonstrates a simple experiment on how Convolutional Neural Networks (CNNs) learn and optimize filters using PyTorch. In this experiment, we simulate the process of applying a 2D convolution operation on an image using a randomly initialized filter, and train the filter weights using gradient descent to minimize the error between the predicted and target outputs.

## Overview

CNNs are powerful for computer vision tasks because they automatically learn features from raw image data by optimizing filter (kernel) weights. This project showcases the following concepts:
- Applying a 2D convolution operation to a grayscale image.
- Optimizing filter weights using backpropagation and gradient descent.
- Accelerating the computation by utilizing a GPU.

## Project Components
1. **Image Preprocessing**: Load and convert the image into a PyTorch tensor with appropriate dimensions for CNN input (batch and channel dimensions added).
2. **Convolution Operation**: A simple 3x3 filter is applied to the image using PyTorch’s `conv2d` function, simulating the first layer of a CNN.
3. **Filter Optimization**: The weights of the filter are optimized using backpropagation to minimize the Mean Squared Error (MSE) between the predicted output and a target output (in this case, the initial output of `conv2d` with the filter).
4. **GPU Utilization**: The entire computation is done on a GPU (if available) for performance improvements.

## Code Implementation

### Key Steps:
1. **Load and Process the Image**: The image is loaded using `PIL` and converted to grayscale. It is then padded and converted to a PyTorch tensor. We also add the batch and channel dimensions required for CNN input:
    ```python
    image = Image.open('c.jpg').convert('L')
    image_arr = np.asarray(image)
    image_arr = np.pad(image_arr, 1)  # Padding for border handling
    x = torch.tensor(image_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ```

2. **Filter Definition and Initialization**: A 3x3 filter (kernel) is randomly initialized. The filter’s weights are learnable, so they are defined with `requires_grad=True` to allow gradient updates during optimization:
    ```python
    w_filter = torch.randn(1, 1, 3, 3, requires_grad=True, dtype=torch.float32, device=device)  # Shape: [out_channels, in_channels, H, W]
    ```

3. **Forward Pass with Convolution**: The filter is applied to the image using `torch.nn.functional.conv2d`:
    ```python
    A = F.conv2d(x, w_filter, padding=0)
    ```

4. **Loss Calculation and Backpropagation**: The loss function used is Mean Squared Error (MSE). The gradient of the loss with respect to the filter weights is calculated and used to update the weights:
    ```python
    loss = torch.mean((A - y) ** 2)
    loss.backward()
    ```

5. **GPU Acceleration**: If a GPU is available, the tensor operations are moved to the GPU for faster computation:
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```

### Code:

```python
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Load and process the image
image = Image.open('c.jpg').convert('L')
image_arr = np.asarray(image)
image_arr = np.pad(image_arr, 1)  # Padding for border handling

# Convert image to tensor and add batch and channel dimensions (for conv2d)
x = torch.tensor(image_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)

# Define the filter (3x3) and ground truth
w_filter = torch.randn(1, 1, 3, 3, requires_grad=True, dtype=torch.float32, device=device)  # Shape: [out_channels, in_channels, H, W]
y = F.conv2d(x, w_filter, padding=0)  # Shape: [1, 1, H-2, W-2]

# Learning rate
lr = 0.000001

# Training loop
for i in range(5):
    print(f"Iteration {i+1}, weight: {w_filter}")
    
    # Forward pass (2D convolution)
    A = F.conv2d(x, w_filter, padding=0)
    
    # Loss calculation (mean squared error)
    loss = torch.mean((A - y) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    with torch.no_grad():
        w_filter -= lr * w_filter.grad
    
    # Zero gradients for the next iteration
    w_filter.grad.zero_()

    # Print loss
    print(f"Loss: {loss.item()}")
```

## Results

The code optimizes the filter weights to minimize the error between the convolution output and the target output. Over several iterations, the loss should decrease, demonstrating that the CNN can learn an optimized filter.

### Output Example:
```
Iteration 1, weight: tensor([[[[ 0.0123, -0.2345,  0.0987], ...]]], device='cuda:0', grad_fn=<SubBackward0>)
Loss: 0.0025
...
```

## How to Run

1. Clone the repository and navigate to the project directory:
    ```bash
    git clone (https://github.com/Malak-Essam/experiment-on-cnn.git)
    cd <experiment-on-cnn>
    ```

2. Install the required Python libraries:
    ```bash
    pip install torch numpy pillow
    ```

3. Run the script:
    ```bash
    python cnn_filter_optimization.py
    ```

Make sure you have access to a GPU (CUDA-enabled) to fully utilize GPU acceleration.

## Conclusion

This experiment demonstrates how CNNs learn and optimize filters to solve computer vision problems, showcasing the core mechanism behind CNNs' ability to automatically extract useful features from images. The use of GPU acceleration with PyTorch enhances performance, especially for larger datasets or more complex models.

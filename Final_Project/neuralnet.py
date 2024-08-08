import torch                      # Importing PyTorch
import torch.nn as nn             # Importing the neural network module from PyTorch
import torch.nn.functional as F   # Importing functional module from PyTorch

class SRCNN_model(nn.Module):
    def __init__(self, architecture : str) -> None:
        super(SRCNN_model, self).__init__()  # Calling the parent class constructor

        # Ensure the architecture is valid
        if architecture not in ["915", "935", "955"]:
            raise ValueError("architecture must be 915, 935 or 955")
        k = int(architecture[1])  # Extract the kernel size from the architecture string

        # Patch extraction layer: Convolutional layer with 9x9 kernel, 3 input channels (RGB), and 64 output channels
        self.patch_extraction = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9)
        nn.init.zeros_(self.patch_extraction.bias)  # Initialize biases to zero
        nn.init.normal_(self.patch_extraction.weight, mean=0.0, std=0.001)  # Initialize weights with a normal distribution

        # Non-linear mapping layer: Convolutional layer with kxk kernel, 64 input channels, and 32 output channels
        self.nonlinear_map = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=k)
        nn.init.zeros_(self.nonlinear_map.bias)  # Initialize biases to zero
        nn.init.normal_(self.nonlinear_map.weight, mean=0.0, std=0.001)  # Initialize weights with a normal distribution

        # Reconstruction layer: Convolutional layer with 5x5 kernel, 32 input channels, and 3 output channels (RGB)
        self.recon = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5)
        nn.init.zeros_(self.recon.bias)  # Initialize biases to zero
        nn.init.normal_(self.recon.weight, mean=0.0, std=0.001)  # Initialize weights with a normal distribution

    def forward(self, X_in):
        # Apply ReLU activation after the patch extraction layer
        X = F.relu(self.patch_extraction(X_in))
        # Apply ReLU activation after the non-linear mapping layer
        X = F.relu(self.nonlinear_map(X))
        # Apply the reconstruction layer
        X = self.recon(X)
        # Clip the output values to the range [0, 1]
        X_out = torch.clip(X, 0.0, 1.0)
        return X_out
from utils.common import *       # Importing all functions from utils.common
from model import SRCNN          # Importing the SRCNN model from the model module
import torch                     # Importing PyTorch
import argparse                  # Importing argparse for command-line argument parsing

# Setting up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int, default=2,     help='Scale factor for super-resolution')
parser.add_argument('--architecture', type=str, default="915", help='Model architecture version')
parser.add_argument('--ckpt-path',    type=str, default="",    help='Path to the model checkpoint')

# -----------------------------------------------------------
# Global variables
# -----------------------------------------------------------

# Parsing command-line arguments
FLAGS, unparsed = parser.parse_known_args()
scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")  # Ensure the scale is valid

architecture = FLAGS.architecture
if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935, or 955")  # Ensure the architecture is valid

# Setting the checkpoint path
ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/SRCNN{architecture}/SRCNN-{architecture}.pt"

# Setting the sigma value for Gaussian blur based on the scale factor
sigma = 0.3 if scale == 2 else 0.2
# Calculating the padding value based on the architecture
pad = int(architecture[1]) // 2 + 6

# -----------------------------------------------------------
# Test function
# -----------------------------------------------------------
def main():
    # Determine the device to use (CUDA for GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the SRCNN model with the specified architecture and device
    model = SRCNN(architecture, device)
    model.load_weights(ckpt_path)  # Load the model weights from the checkpoint

    # Get sorted lists of test data and labels
    ls_data = sorted_list(f"dataset/test/x{scale}/data")
    ls_labels = sorted_list(f"dataset/test/x{scale}/labels")

    sum_psnr = 0  # Initialize the sum of PSNR values
    with torch.no_grad():  # Disable gradient calculation for inference
        for i in range(0, len(ls_data)):
            lr_image = read_image(ls_data[i])             # Read the low-resolution image
            lr_image = gaussian_blur(lr_image, sigma=sigma)  # Apply Gaussian blur to the image
            bicubic_image = upscale(lr_image, scale)      # Upscale the image using bicubic interpolation
            hr_image = read_image(ls_labels[i])           # Read the high-resolution image

            # Convert images from RGB to YCbCr color space
            bicubic_image = rgb2ycbcr(bicubic_image)
            hr_image = rgb2ycbcr(hr_image[:, pad:-pad, pad:-pad])

            # Normalize the images to [0, 1] range
            bicubic_image = norm01(bicubic_image)
            hr_image = norm01(hr_image)

            # Prepare the input image for the model
            bicubic_image = torch.unsqueeze(bicubic_image, dim=0).to(device)
            sr_image = model.predict(bicubic_image)[0].cpu()  # Get the super-resolved image from the model

            # Calculate PSNR and add to the sum
            sum_psnr += PSNR(hr_image, sr_image, max_val=1)

    # Print the average PSNR over all test images
    print(sum_psnr.numpy() / len(ls_data))

if __name__ == "__main__":
    main()  # Execute the main function
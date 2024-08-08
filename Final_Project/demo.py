from utils.common import *  # Import all utility functions from utils.common
from model import SRCNN     # Import the SRCNN class from the model module
import argparse             # Import argparse for parsing command line arguments

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int,   default=2,                   help='-')  # Scale factor for image upscaling
parser.add_argument("--ckpt-path",    type=str,   default="",                  help='-')  # Path to the checkpoint file
parser.add_argument('--architecture', type=str,   default="915",               help='-')  # Architecture type for the model
parser.add_argument("--image-path",   type=str,   default="dataset/test1.png", help='-')  # Path to the input image

# Parse the command line arguments
FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path  # Get the input image path

# Validate the architecture argument
architecture = FLAGS.architecture
if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935, 955")

# Validate the scale argument
scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3 or 4")

# Set the checkpoint path
ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/SRCNN{architecture}/SRCNN-{architecture}.pt"

# Set the sigma value based on the scale
sigma = 0.3 if scale == 2 else 0.2
pad = int(architecture[1]) // 2 + 6  # Padding value based on the architecture

# -----------------------------------------------------------
# demo
# -----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Select device (GPU or CPU)

    # Read the input image
    lr_image = read_image(image_path)
    # Upscale the image using bicubic interpolation
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = bicubic_image[:, pad:-pad, pad:-pad]  # Remove padding
    write_image("bicubic.png", bicubic_image)  # Save the bicubic image

    # Apply Gaussian blur to the low-resolution image
    lr_image = gaussian_blur(lr_image, sigma=sigma)
    # Upscale the blurred image
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = rgb2ycbcr(bicubic_image)  # Convert image to YCbCr color space
    bicubic_image = norm01(bicubic_image)     # Normalize the image
    bicubic_image = torch.unsqueeze(bicubic_image, dim=0)  # Add batch dimension

    # Initialize the SRCNN model
    model = SRCNN(architecture, device)
    # Load the model weights from the checkpoint
    model.load_weights(ckpt_path)
    with torch.no_grad():
        bicubic_image = bicubic_image.to(device)  # Move the image to the selected device
        sr_image = model.predict(bicubic_image)[0]  # Perform super-resolution prediction

    # Denormalize the super-resolved image
    sr_image = denorm01(sr_image)
    sr_image = sr_image.type(torch.uint8)  # Convert to uint8 type
    sr_image = ycbcr2rgb(sr_image)         # Convert back to RGB color space

    # Save the super-resolved image
    write_image("sr.png", sr_image)

if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly
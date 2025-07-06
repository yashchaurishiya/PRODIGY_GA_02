# PRODIGY_GA_02
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Set device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained Stable Diffusion model (requires internet for first run)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# Optional: Authenticate if required (e.g., for some hosted models)
# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token="your_hf_token")

# Define your text prompt
prompt = "A futuristic cityscape at night with flying cars and neon lights"

# Generate the image
with torch.no_grad():  # Disable gradient calculation for efficiency
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# Save or display the generated image
image.save("generated_image.png")
print("Image saved as generated_image.png")

# Optionally display the image
image.show()

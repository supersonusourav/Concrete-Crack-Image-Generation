import torch
from diffusers import StableDiffusionPipeline
import os

def generate_base_image():
    # Minimal Change: Check for GPU availability and prompt user if missing
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        print("\n[WARNING] GPU (CUDA) not detected on this machine.")
        user_choice = input("Would you like to fall back and use the CPU to generate the image? (y/n): ").strip().lower()
        if user_choice != 'y':
            print("Execution aborted by user. Please run on a machine with an NVIDIA GPU.")
            return
        print("Proceeding with CPU generation... (Note: This will take a few minutes)\n")
        device = "cpu"
        dtype = torch.float32  # CPU requires float32 to prevent numerical issues
        
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    
    print("Loading raw, un-tuned Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,  # Dynamic assignment based on device
        use_safetensors=True
    )
    pipe.to(device)  # Dynamic routing to cuda or cpu
    
    # Define a clean prompt matching your training dataset's textual concept
    prompt = "Hyperrealistic close-up photograph of building wall cracks, detailed fissures on concrete and plaster surfaces, natural lighting, high-resolution texture, civil engineering inspection style, sharp focus, realistic depth, authentic material wear, professional documentation photo"
    negative_prompt = "cartoon, illustration, painting, abstract, artistic style, unrealistic, smooth surfaces, blurred, low resolution, watermark, text, logo, extra objects, people, animals, plants, fantasy, exaggerated damage, artificial patterns, digital artifacts"
    
    # Dynamic device routing applied to the random generator anchor
    generator = torch.Generator(device=device).manual_seed(42)
    
    print("Generating image using stock weights...")
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,      # 30 steps is standard for high-quality generation
        guidance_scale=7.5,          # Controls how strictly the model follows the prompt
        generator=generator
    )
    
    # Save the output image
    os.makedirs("./outputs", exist_ok=True)
    image_path = "./outputs/base_sd15_output.png"
    output.images[0].save(image_path)
    print(f"Success! Base image saved completely to: {image_path}")

if __name__ == "__main__":
    generate_base_image()
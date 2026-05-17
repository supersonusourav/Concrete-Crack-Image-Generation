<<<<<<< HEAD
import torch
from diffusers import StableDiffusionPipeline
import os

def generate_lora_image():
    # Check for GPU availability and prompt user if missing
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
        
    base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_weights_dir = "./sd-concrete-crack-lora/checkpoint-1500"
    
    print("Loading base Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, 
        torch_dtype=dtype,  # Dynamic assignment based on device
        use_safetensors=True
    )
    pipe.to(device)  # Dynamic routing to cuda or cpu
    
    print(f"Injecting custom LoRA layers from: {lora_weights_dir}...")
    pipe.load_lora_weights(lora_weights_dir)
    
    prompt = "Hyperrealistic close-up photograph of building wall cracks, detailed fissures on concrete and plaster surfaces, natural lighting, high-resolution texture, civil engineering inspection style, sharp focus, realistic depth, authentic material wear, professional documentation photo"
    negative_prompt = "cartoon, illustration, painting, abstract, artistic style, unrealistic, smooth surfaces, blurred, low resolution, watermark, text, logo, extra objects, people, animals, plants, fantasy, exaggerated damage, artificial patterns, digital artifacts"
    
    # Dynamic device routing applied to the random generator anchor
    generator = torch.Generator(device=device).manual_seed(42)
    
    print("Generating image using fine-tuned weights...")
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
        cross_attention_kwargs={"scale": 0.58}
    )
    
    os.makedirs("./outputs", exist_ok=True)
    image_path = "./outputs/lora_tuned_output.png"
    output.images[0].save(image_path)
    print(f"Success! Fine-tuned image saved completely to: {image_path}")

if __name__ == "__main__":
=======
import torch
from diffusers import StableDiffusionPipeline
import os

def generate_lora_image():
    # Check for GPU availability and prompt user if missing
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
        
    base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_weights_dir = "./sd-concrete-crack-lora/checkpoint-1500"
    
    print("Loading base Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, 
        torch_dtype=dtype,  # Dynamic assignment based on device
        use_safetensors=True
    )
    pipe.to(device)  # Dynamic routing to cuda or cpu
    
    print(f"Injecting custom LoRA layers from: {lora_weights_dir}...")
    pipe.load_lora_weights(lora_weights_dir)
    
    prompt = "Hyperrealistic close-up photograph of building wall cracks, detailed fissures on concrete and plaster surfaces, natural lighting, high-resolution texture, civil engineering inspection style, sharp focus, realistic depth, authentic material wear, professional documentation photo"
    negative_prompt = "cartoon, illustration, painting, abstract, artistic style, unrealistic, smooth surfaces, blurred, low resolution, watermark, text, logo, extra objects, people, animals, plants, fantasy, exaggerated damage, artificial patterns, digital artifacts"
    
    # Dynamic device routing applied to the random generator anchor
    generator = torch.Generator(device=device).manual_seed(42)
    
    print("Generating image using fine-tuned weights...")
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
        cross_attention_kwargs={"scale": 0.58}
    )
    
    os.makedirs("./outputs", exist_ok=True)
    image_path = "./outputs/lora_tuned_output.png"
    output.images[0].save(image_path)
    print(f"Success! Fine-tuned image saved completely to: {image_path}")

if __name__ == "__main__":
>>>>>>> 5571769d2f931f673f57032c6633c1233b71f731
    generate_lora_image()
#!/usr/bin/env python3
"""
LoRA Download Helper
Setup and instructions for cinematic LoRA models
"""

import os
from pathlib import Path

def setup_lora_directory():
    """Create loras directory"""
    lora_dir = Path("./loras")
    lora_dir.mkdir(exist_ok=True)
    print(f"LoRA directory ready: {lora_dir.absolute()}")
    return lora_dir

def show_instructions():
    """Display download instructions"""
    print("\n" + "="*60)
    print("LORA DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
To get high-quality cinematic LoRAs:

1. CivitAI (Recommended):
   https://civitai.com
   
   Search for:
   - "cinematic"
   - "film grain" 
   - "photography"
   - "realistic lighting"
   
   Download .safetensors files to: ./loras/

2. Hugging Face:
   https://huggingface.co/models?other=lora
   
   Filter: Stable Diffusion, LoRA
   Look for: Cinematic, Film, Photography tags

RECOMMENDED LORAS:
- Cinematic Film Lighting - Professional lighting
- 35mm Film Grain - Authentic film texture
- Anamorphic Bokeh - Cinematic depth of field
- Film Noir - Classic noir aesthetic

USAGE:
1. Download .safetensors file
2. Place in ./loras/ directory
3. Update config.yaml:
   lora_path: "./loras/your_file.safetensors"
4. Run: python generate.py

IMPORTANT:
- Only use LoRAs compatible with SD 1.5
- File size typically 10-150MB
- .safetensors format is recommended
""")
    print("="*60 + "\n")

def create_config_example():
    """Create example configuration"""
    config_example = """
# Add this to your config.yaml:

diffusion:
  use_lora: true
  lora_path: "./loras/cinematic_v1.safetensors"
  lora_scale: 0.7
  
  style_tokens:
    - "cinematic lighting"
    - "film grain"
    - "35mm photograph"
    - "professional color grading"
    - "bokeh background"
"""
    
    config_path = Path("./loras/config_example.txt")
    config_path.write_text(config_example)
    print(f"Configuration example saved to: {config_path}")

def main():
    print("Rio Bravo LoRA Setup\n")
    
    lora_dir = setup_lora_directory()
    show_instructions()
    create_config_example()
    
    print("Setup complete!")
    print(f"Place your LoRA files in: {lora_dir}")
    print("Then run: python generate.py\n")

if __name__ == "__main__":
    main()
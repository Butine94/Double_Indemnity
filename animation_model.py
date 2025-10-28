import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video
from PIL import Image
import os
from typing import List, Dict

class CharacterAnimationModel:
    """AnimateDiff with character consistency"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.character_image = None
        self._load_model()
    
    def _load_model(self):
        """Load AnimateDiff with IP-Adapter"""
        print(f"Loading AnimateDiff on {self.device}...")
        
        dtype = torch.float32 if self.device == "cpu" else (
            torch.float16 if self.config['diffusion']['dtype'] == 'fp16' else torch.float32
        )
        
        adapter = MotionAdapter.from_pretrained(
            self.config['animation']['motion_module'],
            torch_dtype=dtype
        )
        
        self.pipe = AnimateDiffPipeline.from_pretrained(
            self.config['diffusion']['base_model'],
            motion_adapter=adapter,
            torch_dtype=dtype
        )
        
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False
        )
        
        self.pipe = self.pipe.to(self.device)
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        
        
        if self.config['character']['use_ip_adapter']:
            self._load_character()
        
        print("Model loaded")

    def _load_character(self):
        """Load character reference"""
        ref_path = self.config['character']['reference_image']
        
        if not os.path.exists(ref_path):
            print(f"Warning: Character reference not found: {ref_path}")
            return
        
        try:
            self.pipe.load_ip_adapter(
                self.config['character']['ip_adapter_model'],
                subfolder="models",
                weight_name=self.config['character']['ip_adapter_weight']
            )
            
            self.character_image = Image.open(ref_path).convert("RGB").resize((512, 512))
            self.pipe.set_ip_adapter_scale(self.config['character']['scale'])
            print(f"Character loaded: {ref_path}")
        except Exception as e:
            print(f"Character loading failed: {e}")
    
    def generate_shots(self, shots: List[Dict], output_dir: str) -> List[Dict]:
        """Generate animated clips"""
        if not self.pipe:
            raise RuntimeError("Model not loaded")
        
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config['diffusion']['seed'])
        
        updated_shots = []
        
        for i, shot in enumerate(shots):
            print(f"Generating shot {i+1}/{len(shots)}: {shot['shot_type']}")
            
            prompt = f"{shot['prompt']}, cinematic film still, smooth motion, atmospheric lighting, film noir, highly detailed"
            
            gen_kwargs = {
                'prompt': prompt,
                'negative_prompt': "static, blurry, low quality, cartoon",
                'num_frames': self.config['animation']['num_frames'],
                'height': self.config['diffusion']['height'],
                'width': self.config['diffusion']['width'],
                'num_inference_steps': self.config['animation']['num_inference_steps'],
                'guidance_scale': self.config['animation']['guidance_scale'],
                'generator': generator
            }
            
            if self.character_image is not None:
                gen_kwargs['ip_adapter_image'] = self.character_image
            
            with torch.no_grad():
                frames = self.pipe(**gen_kwargs).frames[0]
            
            video_path = os.path.join(output_dir, f"shot_{i+1}.mp4")
            export_to_video(frames, video_path, fps=self.config['animation']['fps'])
            
            shot['video_path'] = video_path
            updated_shots.append(shot)
            
            print(f"Saved: {video_path}")
        
        return updated_shots
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.pipe:
            del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
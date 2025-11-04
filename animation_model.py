import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
import os
from typing import List, Dict
from compel import Compel

class CharacterAnimationModel:
    """Hybrid SD + AnimateDiff with character consistency"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.sd_pipe = None
        self.compel = None
        self.sd_compel = None
        self.character_image = None
        self._load_model()
    
    def _load_model(self):
        """Load both SD and AnimateDiff pipelines"""
        print(f"Loading models on {self.device}...")
        
        dtype = torch.float32 if self.device == "cpu" else (
            torch.float16 if self.config['diffusion']['dtype'] == 'fp16' else torch.float32
        )
        
        # Load Stable Diffusion for high-quality stills
        print("Loading Stable Diffusion for keyframes...")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            self.config['diffusion']['base_model'],
            torch_dtype=dtype
        )
        self.sd_pipe = self.sd_pipe.to(self.device)
        
        if self.device == "cuda":
            self.sd_pipe.enable_vae_slicing()
            self.sd_pipe.enable_attention_slicing(slice_size="auto")
        
        self.sd_compel = Compel(
            tokenizer=self.sd_pipe.tokenizer,
            text_encoder=self.sd_pipe.text_encoder
        )
        
        # Load AnimateDiff for motion
        print("Loading AnimateDiff for motion...")
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
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            self.pipe.enable_attention_slicing(slice_size="auto")
        
        self.compel = Compel(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder
        )
        
        if self.config['character']['use_ip_adapter']:
            self._load_character()
        
        print("Models loaded")

    def _load_character(self):
        """Load character reference for both pipelines"""
        ref_path = self.config['character']['reference_image']
        
        if not os.path.exists(ref_path):
            print(f"Warning: Character reference not found: {ref_path}")
            return
        
        try:
            # Load IP-Adapter for AnimateDiff
            self.pipe.load_ip_adapter(
                self.config['character']['ip_adapter_model'],
                subfolder="models",
                weight_name=self.config['character']['ip_adapter_weight']
            )
            
            if hasattr(self.pipe, 'image_encoder') and self.pipe.image_encoder is not None:
                self.pipe.image_encoder = self.pipe.image_encoder.to(self.device)
            
            self.pipe.set_ip_adapter_scale(self.config['character']['scale'])
            
            # Load IP-Adapter for SD
            self.sd_pipe.load_ip_adapter(
                self.config['character']['ip_adapter_model'],
                subfolder="models",
                weight_name=self.config['character']['ip_adapter_weight']
            )
            
            if hasattr(self.sd_pipe, 'image_encoder') and self.sd_pipe.image_encoder is not None:
                self.sd_pipe.image_encoder = self.sd_pipe.image_encoder.to(self.device)
            
            self.sd_pipe.set_ip_adapter_scale(self.config['character']['scale'])
            
            # Load character image
            self.character_image = Image.open(ref_path).convert("RGB").resize((512, 512))
            print(f"Character loaded: {ref_path}")
            
        except Exception as e:
            print(f"Character loading failed: {e}")

    def generate_sd_still(self, prompt: str, negative: str, seed: int) -> Image.Image:
        """Generate high-quality still with Stable Diffusion"""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        gen_kwargs = {
            'prompt_embeds': self.sd_compel(prompt),
            'negative_prompt_embeds': self.sd_compel(negative),
            'height': self.config['diffusion']['height'],
            'width': self.config['diffusion']['width'],
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'generator': generator
        }
        
        if self.character_image is not None:
            gen_kwargs['ip_adapter_image'] = self.character_image
        
        with torch.no_grad():
            image = self.sd_pipe(**gen_kwargs).images[0]
        
        return image

    def generate_shots(self, shots: List[Dict], output_dir: str) -> List[Dict]:
        """Generate shots using hybrid SD + AnimateDiff approach"""
        if not self.pipe or not self.sd_pipe:
            raise RuntimeError("Models not loaded")
        
        base_style = "film noir 1940s detective, grayscale, steady locked camera, consistent character model, same person, frontal view, centered composition"
        negative_base = "color, flickering, morphing, warping, distortion, inconsistent character, multiple faces, duplicate limbs, unstable, blurry, frame jumps"
        
        updated_shots = []
        
        for i, shot in enumerate(shots):
            print(f"\n{'='*60}")
            print(f"Generating shot {i+1}/{len(shots)}: {shot['shot_type']}")
            print(f"Method: {shot.get('method', 'animatediff')}")
            print(f"{'='*60}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            prompt = f"{base_style}, {shot['prompt']}, photorealistic, sharp detail"
            negative = negative_base
            
            method = shot.get('method', 'animatediff')
            
            if method == 'sd_still':
                print("Generating high-quality still with SD...")
                image = self.generate_sd_still(prompt, negative, self.config['diffusion']['seed'] + i)
                
                video_path = os.path.join(output_dir, f"shot_{i+1}.mp4")
                frames = [image] * self.config['animation']['num_frames']
                export_to_video(frames, video_path, fps=self.config['animation']['fps'])
                
            else:
                print("Generating motion with AnimateDiff...")
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self.config['diffusion']['seed'] + i)
                
                gen_kwargs = {
                    'prompt_embeds': self.compel(prompt),
                    'negative_prompt_embeds': self.compel(negative),
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return updated_shots
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.pipe:
            del self.pipe
        if self.sd_pipe:
            del self.sd_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
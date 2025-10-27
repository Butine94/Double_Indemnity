"""
Double Indemnity - Animated Film Generation
Character-driven sequences with AnimateDiff
"""

import yaml
import argparse
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from animation_model import CharacterAnimationModel
from utils.text_processing import parse_script
from utils.video_utils import compile_sequence, setup_directories, save_metadata, read_text_file

def main():
    """Main pipeline execution"""
    
    parser = argparse.ArgumentParser(description='Generate animated film sequences')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--script', default=None, help='Script file path')
    parser.add_argument('--reference', default=None, help='Character reference image')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--num-shots', type=int, default=None, help='Number of shots')
    parser.add_argument('--fps', type=int, default=None, help='Animation FPS')
    args = parser.parse_args()
    
    print("Loading configuration...")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    script_path = args.script if args.script else config['script']['screenplay_path']
    output_dir = args.output if args.output else config['output']['directory']
    num_shots = args.num_shots if args.num_shots else config['scene']['max_shots']
    
    if args.reference:
        config['character']['reference_image'] = args.reference
    if args.fps:
        config['animation']['fps'] = args.fps
    
    setup_directories([output_dir])
    
    print(f"Reading script from {script_path}...")
    script_text = read_text_file(script_path)
    
    if not script_text:
        print("Error: Could not read script file")
        return
    
    print(f"Parsing script into {num_shots} animated shots...")
    scenes = parse_script(script_text, num_scenes=num_shots)
    
    print(f"\nGenerated {len(scenes)} scenes:")
    for scene in scenes:
        print(f"  Shot {scene['id']}: {scene['shot_type']}")
        print(f"    {scene['prompt'][:80]}...")
    
    print("\nInitializing AnimateDiff model...")
    model = CharacterAnimationModel(config)
    
    print(f"\nGenerating {len(scenes)} animated clips...")
    print(f"  Resolution: {config['diffusion']['width']}x{config['diffusion']['height']}")
    print(f"  Frames: {config['animation']['num_frames']}")
    print(f"  FPS: {config['animation']['fps']}")
    print(f"  Character: {'Enabled' if config['character']['use_ip_adapter'] else 'Disabled'}\n")
    
    generated_scenes = model.generate_shots(scenes, output_dir)
    
    metadata = {
        'config': config,
        'script_path': script_path,
        'num_shots': len(generated_scenes),
        'scenes': generated_scenes
    }
    save_metadata(metadata, str(Path(output_dir) / 'metadata.json'))
    
    print(f"\nGeneration complete!")
    print(f"  Output: {output_dir}")
    print(f"  Clips: {len(generated_scenes)}")
    
    print("\nCompiling final sequence...")
    clip_paths = [scene['video_path'] for scene in generated_scenes]
    final_path = Path(output_dir) / 'double_indemnity_sequence.mp4'
    
    if compile_sequence(clip_paths, str(final_path), config['output']['final_fps']):
        print(f"Final video: {final_path}")
    else:
        print("Sequence compilation failed")
    
    model.cleanup()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
Rio Bravo
AI film generator that converts screenplay text into cinematic storyboards using Stable Diffusion with ControlNet composition control and LoRA style consistency.

Features:
ControlNet Integration - Spatial composition control for consistent framing and depth
LoRA Style Application - Cinematic aesthetics (film grain, lighting, color grading)
Script Parsing - Extracts visual elements from screenplay text
Professional Quality - High-resolution outputs with optimized inference
Video Compilation - Optional MP4 generation from image sequences


# Install dependencies
pip install -r requirements.txt

# Generate storyboards from script
python generate.py

# With video output
python generate.py --video

# Custom settings
python generate.py --script data/input.txt --num-shots 10 --depth-consistency


Example Output
[Add 2-3 generated storyboard images here]
Generated from the Rio Bravo script: Mojave Desert trains sequence with cinematic composition and film aesthetic.

Rio_Bravo/
├── generate.py              # Main pipeline
├── diffusion_model.py       # Core model with ControlNet/LoRA
├── config.yaml              # Configuration settings
├── data/
│   └── input.txt           # Script input
├── utils/
│   ├── text_processing.py  # Script parsing
│   ├── io_utils.py         # File operations
│   └── controlnet_utils.py # Preprocessing
└── outputs/                 # Generated images



Double Indemnity
AI film generator that creates animated character-driven sequences from screenplay text using AnimateDiff with character consistency and temporal coherence.

Features
AnimateDiff Integration - Generate 2-3 second animated video clips, not just static images
Character Consistency - Same protagonist across all shots using IP-Adapter
Temporal Coherence - Smooth motion with frame interpolation
Script Parsing - Extracts character actions and camera movements
Video Compilation - Automatic sequence assembly with transitions


# Install dependencies
pip install -r requirements.txt

# Generate animated sequence
python generate.py

# Custom character reference
python generate.py --reference ./my_character.jpg

# With custom script
python generate.py --script data/input.txt --fps 16

# Generate sequence with specific number of shots
python generate.py --num-shots 5


Example Output
*[Add generated animation examples here]*
Generated from noir-style script: lone figure in rain-soaked alley with consistent character and smooth motion.


Double_Indemnity/
├── generate.py              # Main pipeline
├── animation_model.py       # AnimateDiff with IP-Adapter
├── config.yaml              # Configuration
├── data/
│   └── input.txt           # Script input
├── references/
│   └── protagonist.png     # Character reference image
├── utils/
│   ├── text_processing.py  # Script parsing with motion detection
│   └── video_utils.py      # Video assembly and effects
└── outputs/                 # Generated animations

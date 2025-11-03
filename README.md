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

"""
Text processing with motion detection
"""

import re
from typing import List, Dict

SHOT_TYPES = ['wide establishing', 'medium tracking', 'close-up character', 'detail insert', 'dramatic reveal']
MOTION_KEYWORDS = ['steps', 'walks', 'moves', 'pauses', 'turns']

def parse_script(text: str, num_scenes: int = 5) -> List[Dict]:
    """Parse script into animated scenes"""
    sentences = split_into_segments(text)
    
    if not sentences:
        return generate_fallback_scenes(num_scenes)
    
    return [
        {
            'id': i + 1,
            'shot_type': SHOT_TYPES[i % len(SHOT_TYPES)],
            'prompt': sentences[i] if i < len(sentences) else f"continuation, {SHOT_TYPES[i % len(SHOT_TYPES)]}"
        }
        for i in range(num_scenes)
    ]


def split_into_segments(text: str) -> List[str]:
    """Split text into sentences"""
    text = re.sub(r'\s+', ' ', text).strip()
    segments = re.split(r'[.!?]+', text)
    return [s.strip() for s in segments if len(s.strip()) > 15]


def generate_fallback_scenes(num_scenes: int) -> List[Dict]:
    """Generate default scenes"""
    fallbacks = [
        "lone figure in rain-soaked alley, noir atmosphere",
        "neon reflections on wet pavement, moody lighting",
        "character silhouette against flickering lamp",
        "steam rising from grates, urban mystery",
        "narrow passage leading to distant light"
    ]
    
    return [
        {
            'id': i + 1,
            'shot_type': SHOT_TYPES[i % len(SHOT_TYPES)],
            'prompt': fallbacks[i % len(fallbacks)]
        }
        for i in range(num_scenes)
    ]
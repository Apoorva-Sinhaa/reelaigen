import sys
import os

# Add parent directory to path so imports work when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.manim_model import ManimModel
from models.voice_model import VoiceModel
import json
from typing import List, Dict

def main():
   manim_model = ManimModel()
   example_text = "In this video, we will explore the fascinating world of mathematics. We will start with an introduction to basic concepts, followed by detailed explanations and visualizations of complex equations. Finally, we will summarize the key points and provide additional resources for further learning."
   manim_data = manim_model.generate_manim_data(content_text=example_text, time_length=300)

   print("Generated Manim Data:")
   print(manim_data)


def generate_audio_from_reels():
   with open("output/generated_reels.json", "r") as f:
      reel_data = json.load(f)
   
   # Extract the reels array from the JSON structure
   reels = reel_data.get("reels", [])
   
   if not reels:
      print("No reels found in JSON file")
      return
   
   voice_model = VoiceModel()
   audio_output_path = "final_output.mp3"
   voice_model.generate_audio_from_reels(reels, output_path=audio_output_path, speed=1.0)
   print(f"Audio saved to {audio_output_path}")


if __name__ == "__main__":
   generate_audio_from_reels()



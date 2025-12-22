import os
# Set all Hugging Face cache directories BEFORE importing transformers
# This ensures all downloads go to G: drive instead of C: drive
cache_base = r"E:\huggingface_cache"
os.environ["HF_HOME"] = cache_base
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_base, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "models")

from models.manim_model import ManimModel
import torch 
def main():
   manim_model = ManimModel()
   example_text = "In this video, we will explore the fascinating world of mathematics. We will start with an introduction to basic concepts, followed by detailed explanations and visualizations of complex equations. Finally, we will summarize the key points and provide additional resources for further learning."
   manim_data = manim_model.generate_manim_data(content_text=example_text, time_length=300)

   print("Generated Manim Data:")
   print(manim_data)
    # print("Torch is successfully imported. Version:", torch.__version__)
    # print("CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()

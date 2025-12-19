from models.manim_model import ManimModel

def main():
   manim_model = ManimModel()
   example_text = "In this video, we will explore the fascinating world of mathematics. We will start with an introduction to basic concepts, followed by detailed explanations and visualizations of complex equations. Finally, we will summarize the key points and provide additional resources for further learning."
   manim_data = manim_model.generate_manim_data(content_text=example_text, time_length=300)

   print("Generated Manim Data:")
   print(manim_data)

if __name__ == "__main__":
    main()

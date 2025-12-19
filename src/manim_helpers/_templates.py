import manim

def manim_text_intro(content: str, timestamp: float) -> manim.Text:
    return manim.Text(f"Intro: {content}")



template_map = {
    'text_intro(content, timestamp)': manim_text_intro,
}
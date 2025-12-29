import json
import manim
from pathlib import Path
from manim_helpers._templates import (
    manim_text_intro,
    manim_text_outro,
    manim_bullet_points,
    manim_image_display,
    manim_equation_display,
    manim_step_by_step,
    manim_graph_plot,
    manim_highlight_text,
    manim_transformation,
    manim_definition_box,
    manim_proof_steps,
    manim_comparison,
)


TEMPLATE_MAP = {
    'text_intro': manim_text_intro,
    'text_outro': manim_text_outro,
    'bullet_points': manim_bullet_points,
    'image_display': manim_image_display,
    'equation_display': manim_equation_display,
    'step_by_step': manim_step_by_step,
    'graph_plot': manim_graph_plot,
    'highlight_text': manim_highlight_text,
    'transformation': manim_transformation,
    'definition_box': manim_definition_box,
    'proof_steps': manim_proof_steps,
    'comparison': manim_comparison,
}


def convert_content_to_string(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return '\n'.join(str(item) for item in content)
    elif isinstance(content, dict):
        if 'left' in content and 'right' in content:
            return f"{content['left']}|{content['right']}"
        elif 'equation' in content:
            return content['equation']
        elif 'from' in content and 'to' in content:
            return f"{content['from']} -> {content['to']}"
        else:
            return str(content)
    else:
        return str(content)


class ManimSceneGenerator(manim.Scene):
    def __init__(self, json_path: str = None, **kwargs):
        super().__init__(**kwargs)
        if json_path is None:
            json_path = Path(__file__).parent / "output" / "example_manim_json.json"
        self.json_path = str(json_path)
        self.template_data = self._load_json()
        self.current_objects = []  
    
    def _load_json(self):
        json_path = Path(self.json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def construct(self):
        for item in self.template_data:
            template_name = item.get('templateName')
            content = item.get('content')
            start_time = item.get('startTime', 0)
            end_time = item.get('endTime', 0)
            timestamp = item.get('timestamp', start_time)
            duration = end_time - start_time
            
            if duration <= 0:
                continue
            
            if template_name not in TEMPLATE_MAP:
                print(f"Warning: Template '{template_name}' not found in TEMPLATE_MAP")
                continue
            
            template_func = TEMPLATE_MAP[template_name]
            
            content_str = convert_content_to_string(content)
            
            try:
                manim_obj = template_func(content_str, timestamp)
            except Exception as e:
                print(f"Error creating object for template '{template_name}': {e}")
                continue
            
            if self.current_objects:
                self.play(manim.FadeOut(*self.current_objects), run_time=0.5)
                self.current_objects = []
            
            fade_in_time = min(1.0, duration * 0.3)  
            wait_time = max(0, duration - fade_in_time)
            
            self.play(manim.FadeIn(manim_obj), run_time=fade_in_time)
            
            if wait_time > 0:
                self.wait(wait_time)
            
            self.current_objects = [manim_obj]
        
        if self.current_objects:
            self.play(manim.FadeOut(*self.current_objects), run_time=0.5)


if __name__ == "__main__":
    pass


import gradio as gr
import sys
sys.path.append("./scripts/")

from scripts.convertor import pil2cv, cv2pil, df2bgra
from scripts.processor import get_base, get_normal_layer, get_composite_layer
from scripts.utils import save_psd

import cv2
from pytoshop.enums import BlendMode



class webui:
    def __init__(self):
        self.demo = gr.Blocks()
        
    def divide_layer(self, input_image, roop, init_cluster, ciede_threshold, blur_size, layer_mode):
        image = pil2cv(input_image)
        self.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        df = get_base(self.input_image, roop, init_cluster, ciede_threshold, blur_size, False)        
        
        base_image = cv2pil(df2bgra(df))
        image = cv2pil(image)
        if layer_mode == "composite":
            base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(self.input_image, df)
            filename = save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
                ["base", "screen", "multiply", "subtract", "addition"],
                [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge]
            )
            base_layer_list = [cv2pil(layer) for layer in base_layer_list]
            return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
        elif layer_mode == "normal":
            base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(self.input_image, df)
            filename = save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list],
                ["base", "bright", "shadow"],
                [BlendMode.normal, BlendMode.normal, BlendMode.normal]
            )
            return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
        else:
            return None

        
    
    def get_base_layer_list(self):
        if self.df is None:
            self.divide_layer()

    def launch(self, share):
        with self.demo:
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil")
                    with gr.Box():
                        roop = gr.Slider(1, 20, value=3, step=1, label="roop", show_label=True)
                        init_cluster = gr.Slider(1, 50, value=10, step=1, label="init_cluster", show_label=True)
                        ciede_threshold = gr.Slider(1, 50, value=15, step=1, label="ciede_threshold", show_label=True)
                        blur_size = gr.Slider(1, 20, value=5, label="blur_size", show_label=True)
                        layer_mode = gr.Dropdown(["normal", "composite"], value = "normal", label="output_layer_mode", show_label=True)
                        #split_bg = gr.Checkbox(label="split bg", show_label=True)
                        

                    submit = gr.Button(value="Submit")
                with gr.Row():
                    with gr.Column():
                        with gr.Tab("output"):
                            output_0 = gr.Gallery()
                        with gr.Tab("base"):
                            output_1 = gr.Gallery()
                        with gr.Tab("bright"):
                            output_2 = gr.Gallery()
                        with gr.Tab("shadow"):
                            output_3 = gr.Gallery()

                        output_file = gr.File()
            submit.click(
                self.divide_layer, 
                inputs=[input_image, roop, init_cluster, ciede_threshold, blur_size, layer_mode], 
                outputs=[output_0, output_1, output_2, output_3, output_file]
            )

        self.demo.launch(share)


if __name__ == "__main__":
    ui = webui()
    ui.launch(sys.argv[0])

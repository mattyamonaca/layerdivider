import os
import io
import json
import numpy as np
import cv2

import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks

from scripts.processor import get_base, get_normal_layer, get_composite_layer
from scripts.convertor import pil2cv, cv2pil, df2bgra
from scripts.utils import save_psd
from modules.paths_internal import extensions_dir
from collections import OrderedDict

from pytoshop.enums import BlendMode


model_cache = OrderedDict()
output_dir = os.path.join(
    extensions_dir, "LayerDivider/output/")

class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()

  def title(self):
    return "LayerDivider"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()

def divide_layer(self, input_image, roop, init_cluster, ciede_threshold, blur_size, layer_mode):
    image = pil2cv(input_image)
    self.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    df = get_base(self.input_image, roop, init_cluster, ciede_threshold, blur_size)        
    
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

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as LayerDivider:
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
            divide_layer, 
            inputs=[input_image, roop, init_cluster, ciede_threshold, blur_size, layer_mode], 
            outputs=[output_0, output_1, output_2, output_3, output_file]
        )

    return [(LayerDivider, "LayerDivider", "layerdivider")]

script_callbacks.on_ui_tabs(on_ui_tabs)
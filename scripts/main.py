import os
import io
import json
import numpy as np
import cv2

import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks

from scripts.ld_processor import get_base, get_normal_layer, get_composite_layer
from scripts.ld_convertor import pil2cv, cv2pil, df2bgra
from scripts.ld_utils import save_psd
from modules.paths_internal import extensions_dir
from collections import OrderedDict

from pytoshop.enums import BlendMode


model_cache = OrderedDict()
output_dir = os.path.join(
    extensions_dir, "layerdivider/output")

class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()

  def title(self):
    return "LayerDivider"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()

def divide_layer(input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg):
    image = pil2cv(input_image)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    df = get_base(input_image, loops, init_cluster, ciede_threshold, blur_size, h_split, v_split, n_cluster, alpha, th_rate, split_bg, False) 
    
    base_image = cv2pil(df2bgra(df))
    image = cv2pil(image)
    if layer_mode == "composite":
        base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
            ["base", "screen", "multiply", "subtract", "addition"],
            [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
            output_dir
        )
        base_layer_list = [cv2pil(layer) for layer in base_layer_list]
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    elif layer_mode == "normal":
        base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list],
            ["base", "bright", "shadow"],
            [BlendMode.normal, BlendMode.normal, BlendMode.normal],
            output_dir
        )
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    else:
        return None

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as LayerDivider:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil")
                with gr.Accordion("Layer Settings", open=True):
                    loops = gr.Slider(1, 20, value=1, step=1, label="loops", show_label=True)
                    init_cluster = gr.Slider(1, 50, value=10, step=1, label="init_cluster", show_label=True)
                    ciede_threshold = gr.Slider(1, 50, value=5, step=1, label="ciede_threshold", show_label=True)
                    blur_size = gr.Slider(1, 20, value=5, label="blur_size", show_label=True)
                    layer_mode = gr.Dropdown(["normal", "composite"], value = "normal", label="output_layer_mode", show_label=True)
                    
                with gr.Accordion("BG Settings", open=True):
                    split_bg = gr.Checkbox(label="split bg", show_label=True)
                    h_split = gr.Slider(1, 2048, value=256, step=4, label="horizontal split num", show_label=True)
                    v_split = gr.Slider(1, 2048, value=256, step=4, label="vertical split num", show_label=True)
                    
                    n_cluster = gr.Slider(1, 1000, value=500, step=10, label="cluster num", show_label=True)
                    alpha = gr.Slider(1, 255, value=100, step=1, label="alpha threshold", show_label=True)
                    th_rate = gr.Slider(0, 1, value=0.1, step=0.01, label="mask content ratio", show_label=True)
                    
                    
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
            inputs=[input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg], 
            outputs=[output_0, output_1, output_2, output_3, output_file]
        )

    return [(LayerDivider, "LayerDivider", "layerdivider")]

script_callbacks.on_ui_tabs(on_ui_tabs)
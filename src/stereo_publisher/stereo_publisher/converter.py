import torch
import torch.nn.functional as F
import os
import sys

# --- IMPORT YOUR CUSTOM MODULES ---
# Make sure the folder containing your model definition is in the path
# If your code is in a different folder, uncomment and update the line below:
# sys.path.append('/home/nvidia/ros2_ws/src/your_package/your_package')

from models.get_models import get_model_with_opts
from saver import load_model_for_evaluate
from utils.platform_loader import read_yaml_options

# 1. Setup Paths and Device
# Change these to the actual names of your files
MODEL_PATH = 'model.pth'
CONFIG_PATH = 'exp_opts_working.yaml'
OUTPUT_ONNX = 'stereo_depth.onnx'
device = torch.device('cpu') 

opts_dic = read_yaml_options(CONFIG_PATH)
print("Loading model to CPU...")
depth_network = get_model_with_opts(opts_dic, device)
depth_network = load_model_for_evaluate(MODEL_PATH, depth_network)
depth_network.eval() # DO NOT move to .cuda() yet

# 3. Define the Wrapper for ONNX
# ONNX needs a standard 'forward' method, so we wrap your 'inference_forward'
class DepthWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, color_s, color_o):
        # Re-creating the dictionary format your model expects
        inputs = {
            'color_s': color_s, 
            'color_o': color_o
        }
        outputs = self.model.inference_forward(inputs, is_train=False)
        # We return only the specific tensor we want from the engine
        return outputs['stereo_depth_0_s']

wrapped_model = DepthWrapper(depth_network)
h, w = 462, 630
dummy_left = torch.randn(1, 3, h, w)
dummy_right = torch.randn(1, 3, h, w)

print("Starting export on CPU (to prevent GPU OOM)...")
try:
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model, 
            (dummy_left, dummy_right), 
            OUTPUT_ONNX,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['color_s', 'color_o'],
            output_names=['depth_output']
            # Removed dynamic_axes to simplify the graph for OOM prevention
        )
    print(f"Success! {OUTPUT_ONNX} created.")
except Exception as e:
    print(f"Export failed: {e}")
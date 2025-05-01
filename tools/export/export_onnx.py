import argparse

import torch
from safetensors.torch import load_file
from torch.export import Dim

from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--encoder", type=str, default="vits")
parser.add_argument("--weights", type=str, default="model.safetensors")
parser.add_argument("--opset", type=int, default=17)
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

# Define model encoder kwargs
model_kwargs = dict(
    vits=dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    vitb=dict(
        encoder="vitb",
        features=128,
        out_channels=[96, 192, 384, 768],
    ),
    vitl=dict(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=150.0,
        mode="disparity",
        pretrain_type="dinov2",
        del_mask_token=False,
    ),
)


# Check if onnx is installed
def check_dependencies():
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "onnx is not installed. Please install it using `pip install onnx`."
        )


# Export model to onnx
def export_onnx(
    encoder: str,
    weights_path: str,
    opset: int,
    output_path: str,
):
    model = DepthAnythingV2(**model_kwargs[encoder])

    # Load model weights from safetensors file
    model_weights = load_file(weights_path)
    model.load_state_dict(model_weights)

    # Set model to evaluation mode
    model.eval()

    # Delete model weights from memory
    del model_weights

    # Define dynamic axes
    dynamic_axes = {}
    dynamic_axes[0] = "batch_size"
    dynamic_axes[2] = "height"
    dynamic_axes[3] = "width"

    # Export model to onnx
    # Dynamic input size + no constant folding for TensorRT Conversion
    torch.onnx.export(
        model,
        torch.randn(1, 3, 420, 420),
        output_path,
        opset_version=opset,
        input_names=["image"],
        output_names=["depth"],
        dynamic_axes={"image": dynamic_axes, "depth": dynamic_axes},
    )


if __name__ == "__main__":
    check_dependencies()

    assert args.encoder in model_kwargs, "Invalid encoder"

    if args.output is None:
        args.output = f"distill_any_depth_{args.encoder}.onnx"

    export_onnx(args.encoder, args.weights, args.opset, args.output)

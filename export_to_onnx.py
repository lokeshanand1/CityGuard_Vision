import argparse
from pathlib import Path

from ultralytics import YOLO


def export(best_weights: str, output_dir: str = "onnx_exports") -> Path:
    """
    Export the trained YOLO model to ONNX for ROCm / edge runtimes.

    Example:
      python export_to_onnx.py --weights best_model.pt
    """
    weights_path = Path(best_weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    # Ultralytics will place the ONNX file next to the weights by default.
    result = model.export(format="onnx", opset=12)

    onnx_path = Path(result) if isinstance(result, str) else weights_path.with_suffix(".onnx")
    final_path = output_root / onnx_path.name
    if onnx_path.exists():
        onnx_path.replace(final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="best_model.pt",
        help="Path to YOLOv8 weights file (e.g. best.pt or best_model.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="onnx_exports",
        help="Directory to place exported ONNX model in",
    )
    args = parser.parse_args()

    onnx_path = export(args.weights, args.out_dir)
    print(f"✅ Exported ONNX model to: {onnx_path}")


if __name__ == "__main__":
    main()


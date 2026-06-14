"""
tools/export_onnx.py
Chuyển đổi model YOLO PyTorch (.pt) sang ONNX để chạy với onnxruntime (không cần torch).

Sử dụng:
    python tools/export_onnx.py
    python tools/export_onnx.py --model data/models/best.pt --imgsz 320 --opset 12
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Thêm root vào sys.path để import src.*
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt → .onnx cho dự án face-attendance"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "data" / "models" / "best.pt"),
        help="Đường dẫn file .pt (mặc định: data/models/best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "data" / "models" / "best.onnx"),
        help="Đường dẫn file .onnx đầu ra (mặc định: data/models/best.onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Kích thước ảnh input (mặc định: 320). Phải khớp với SPOOFING_INPUT_SIZE trong config.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (mặc định: 12, tương thích rộng với onnxruntime CPU)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Đơn giản hóa graph ONNX bằng onnx-simplifier (tăng tốc inference)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Export FP16 (chỉ dùng nếu target là GPU — KHÔNG dùng cho CPU inference)",
    )
    return parser.parse_args()


def check_dependencies() -> None:
    """Kiểm tra các thư viện cần thiết trước khi chạy."""
    missing = []
    try:
        import ultralytics
        print(f"  ✅ ultralytics {ultralytics.__version__}")
    except ImportError:
        missing.append("ultralytics")

    try:
        import onnx
        print(f"  ✅ onnx {onnx.__version__}")
    except ImportError:
        missing.append("onnx")

    try:
        import onnxruntime
        print(f"  ✅ onnxruntime {onnxruntime.__version__}")
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print(f"\n❌ Thiếu thư viện: {', '.join(missing)}")
        print(f"   Cài đặt: pip install {' '.join(missing)}")
        sys.exit(1)


def verify_onnx(onnx_path: Path) -> bool:
    """Verify file ONNX hợp lệ bằng onnx.checker."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        return True
    except Exception as exc:
        print(f"  ⚠ ONNX verify thất bại: {exc}")
        return False


def run_test_inference(onnx_path: Path, imgsz: int) -> bool:
    """Chạy thử inference 1 lần để đảm bảo model load được."""
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        inp_name  = sess.get_inputs()[0].name
        out_name  = sess.get_outputs()[0].name
        dummy     = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        output    = sess.run([out_name], {inp_name: dummy})[0]
        print(f"  ✅ Test inference OK — output shape: {output.shape}")
        return True
    except Exception as exc:
        print(f"  ⚠ Test inference thất bại: {exc}")
        return False


def export(args: argparse.Namespace) -> None:
    model_path  = Path(args.model)
    output_path = Path(args.output)

    # ── Kiểm tra file đầu vào ────────────────────────────────────────────
    if not model_path.exists():
        print(f"\n❌ Không tìm thấy file model: {model_path}")
        print("   Đặt file best.pt vào data/models/ trước khi chạy script này.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  FaceAttend — YOLO Export Tool")
    print(f"{'='*55}")
    print(f"  Input  : {model_path}")
    print(f"  Output : {output_path}")
    print(f"  imgsz  : {args.imgsz}x{args.imgsz}")
    print(f"  opset  : {args.opset}")
    print(f"  simplify: {args.simplify}")
    print(f"  half   : {args.half}  {'(FP16 — chỉ cho GPU!)' if args.half else ''}")
    print(f"{'='*55}\n")

    # ── Kiểm tra dependencies ────────────────────────────────────────────
    print("🔍 Kiểm tra dependencies:")
    check_dependencies()

    # ── Load model + export ──────────────────────────────────────────────
    print(f"\n🚀 Đang load model từ: {model_path}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"❌ Không load được model: {exc}")
        sys.exit(1)

    print(f"\n⚙  Đang export sang ONNX (opset={args.opset}, imgsz={args.imgsz})…")
    try:
        exported_path = model.export(
            format   = "onnx",
            imgsz    = args.imgsz,
            opset    = args.opset,
            simplify = args.simplify,
            half     = args.half,
            dynamic  = False,   # Fixed shape → tối ưu hơn cho inference CPU
        )
    except Exception as exc:
        print(f"\n❌ Export thất bại: {exc}")
        sys.exit(1)

    # ultralytics tự đặt tên file .onnx cạnh file .pt
    exported_path = Path(exported_path)
    if not exported_path.exists():
        print(f"❌ File ONNX không được tạo tại: {exported_path}")
        sys.exit(1)

    # ── Di chuyển về đường dẫn đích nếu khác ────────────────────────────
    if exported_path.resolve() != output_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported_path), str(output_path))
        print(f"\n📦 Di chuyển file tới: {output_path}")

    # ── Verify ───────────────────────────────────────────────────────────
    print("\n🔎 Đang verify file ONNX…")
    onnx_ok = verify_onnx(output_path)

    print("\n🧪 Đang chạy test inference…")
    infer_ok = run_test_inference(output_path, args.imgsz)

    # ── Thống kê file ────────────────────────────────────────────────────
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*55}")
    if onnx_ok and infer_ok:
        print(f"  ✅ Export thành công!")
        print(f"  📁 File : {output_path}")
        print(f"  📏 Size : {size_mb:.1f} MB")
        print(f"\n  👉 Cập nhật config.py nếu cần:")
        print(f"     ANTI_SPOOFING_MODEL_PATH = DATA_DIR / 'models' / '{output_path.name}'")
        print(f"     SPOOFING_INPUT_SIZE = ({args.imgsz}, {args.imgsz})")
    else:
        print(f"  ⚠ Export hoàn thành nhưng có cảnh báo.")
        print(f"  📁 File : {output_path}")
        print(f"     Kiểm tra lại model hoặc phiên bản onnxruntime.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    export(parse_args())

import os, argparse, subprocess, sys
from pathlib import Path
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--project', default='runs_yolov5')
    ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--device', default='0')
    ap.add_argument('--yolov5_dir', default='yolov5')
    args = ap.parse_args()
    y5 = Path(args.yolov5_dir)/"train.py"
    if not y5.exists(): raise FileNotFoundError("YOLOv5 repo not found. git clone https://github.com/ultralytics/yolov5")
    cmd = [sys.executable, str(y5), "--img", str(args.imgsz), "--batch", str(args.batch), "--epochs", str(args.epochs), "--data", args.data, "--weights", "yolov5n.pt", "--project", args.project, "--name", "B1_yolov5n", "--device", str(args.device)]
    print(">>>", " ".join(cmd)); subprocess.check_call(cmd)
if __name__ == "__main__": main()

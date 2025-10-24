import argparse
from pathlib import Path
from ultralytics import YOLO
from callbacks.preproc_srts import UltralyticsSRTSCallback
from callbacks.npr_miner import UltralyticsNPRCallback
THIS = Path(__file__).parent.resolve()
def train_once(model_def, data, project, name, imgsz=640, epochs=150, batch=16, device=0, seed=42, callbacks=None):
    model = YOLO(model_def)
    if callbacks:
        for cb in callbacks:
            if hasattr(cb, "on_preprocess_batch"):
                model.add_callback("on_preprocess_batch", cb.on_preprocess_batch)
    return model.train(data=data, imgsz=imgsz, epochs=epochs, batch=batch, device=device, project=project, name=name, seed=seed, workers=4, verbose=True)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--project', default='runs_defects')
    ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--device', default=0)
    args = ap.parse_args()
    model_p2_se = THIS / "models" / "yolov8n_p2_se.yaml"
    txt = model_p2_se.read_text(); (THIS/"models"/"yolov8n_p2_only.yaml").write_text(txt.replace("modules.C2f_SELite","C2f"))
    train_once("yolov8n.pt", args.data, args.project, "B0_yolov8n", imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device)
    train_once(str(THIS/'models/yolov8n_p2_only.yaml'), args.data, args.project, "A1_P2only", imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device)
    train_once(str(model_p2_se), args.data, args.project, "A1A2_P2_SE", imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device)
    cb_srts = UltralyticsSRTSCallback(prob=0.5, sigma=2.5, alpha=0.65)
    train_once(str(model_p2_se), args.data, args.project, "A1A2B1_P2_SE_SRTS", imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device, callbacks=[cb_srts])
    cb_npr = UltralyticsNPRCallback(prob=0.5, patch_size=128)
    train_once(str(model_p2_se), args.data, args.project, "A1A2B1B2_P2_SE_SRTS_NPR", imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device, callbacks=[cb_srts, cb_npr])
if __name__ == "__main__": main()

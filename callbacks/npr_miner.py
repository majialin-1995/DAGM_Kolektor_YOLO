import random, numpy as np, cv2, torch
class NegativePatchReplay:
    def __init__(self, prob=0.5, patch_size=128):
        self.prob, self.ps = prob, patch_size
    def __call__(self, im):
        if random.random() > self.prob: return im
        H,W = im.shape[:2]
        if H < self.ps or W < self.ps: return im
        for _ in range(4):
            x = random.randint(0, W-self.ps); y = random.randint(0, H-self.ps)
            X = random.randint(0, W-self.ps); Y = random.randint(0, H-self.ps)
            patch = im[y:y+self.ps, x:x+self.ps].copy()
            im[Y:Y+self.ps, X:X+self.ps] = cv2.addWeighted(im[Y:Y+self.ps, X:X+self.ps], 0.5, patch, 0.5, 0)
        return im
class UltralyticsNPRCallback:
    def __init__(self, prob=0.5, patch_size=128):
        self.npr = NegativePatchReplay(prob=prob, patch_size=patch_size)

    def on_train_batch_start(self, trainer):
        batch = trainer.batch
        ims = batch["img"]
        ims_np = (ims.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i in range(len(ims_np)):
            ims_np[i] = self.npr(ims_np[i])
        ims_t = (
            torch.from_numpy(ims_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(ims.device)
        )
        batch["img"] = ims_t

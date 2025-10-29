import numpy as np, cv2, torch
from scipy import ndimage
class SRTSPreproc:
    def __init__(self, prob=0.5, sigma=2.5, alpha=0.65):
        self.prob, self.sigma, self.alpha = prob, sigma, alpha
    def saliency_sr(self, gray):
        gray = gray.astype(np.float32)/255.0
        F = np.fft.fft2(gray); A = np.abs(F); L = np.log(A+1e-6)
        R = L - ndimage.gaussian_filter(L, sigma=self.sigma)
        S = np.abs(np.fft.ifft2(np.exp(R + 1j*np.angle(F))))**2
        S = (S - S.min())/(S.max()-S.min()+1e-6); return S
    def __call__(self, im):
        import numpy as np
        if np.random.rand() > self.prob: return im
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        sal = self.saliency_sr(g); sal = cv2.GaussianBlur(sal,(0,0),2.0)
        sal = (sal - sal.min())/(sal.max()-sal.min()+1e-6)
        sal3 = np.repeat(sal[...,None],3,axis=2)
        return (im.astype(np.float32)*(0.5+self.alpha*sal3)).clip(0,255).astype(np.uint8)
class UltralyticsSRTSCallback:
    def __init__(self, prob=0.5, sigma=2.5, alpha=0.65):
        self.t = SRTSPreproc(prob=prob, sigma=sigma, alpha=alpha)

    def on_train_batch_start(self, trainer):
        batch = trainer.batch
        ims = batch["img"]
        ims_np = (ims.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i in range(len(ims_np)):
            ims_np[i] = self.t(ims_np[i])
        ims_t = (
            torch.from_numpy(ims_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(ims.device)
        )
        batch["img"] = ims_t

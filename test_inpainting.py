import imageio
import numpy as np
import matplotlib.pyplot as plt
from thebrain.models.inpainting import Inpainting
from skimage.color import rgb2gray

y0 = 400
y1 = 800
x0 = 850
x1 = 1250
inpainter = Inpainting()
img_orig = imageio.imread('image/moon.jpg')[..., :3]
img_for_inpaint = np.copy(img_orig)
img_for_inpaint[y0:y1, x0:x1, :] = np.array([0, 0, 0], dtype=np.uint8)
msk_for_inpaint = np.zeros(img_for_inpaint.shape[:2], dtype=np.uint8)
msk_for_inpaint[y0:y1, x0:x1] = 255
inpainted = inpainter(image=img_for_inpaint, mask=msk_for_inpaint, use_patchmatch=False)
inpainted = inpainted * 255
inpainted = inpainted.astype(np.uint8)
inpainted_pm = inpainter(image=img_for_inpaint, mask=msk_for_inpaint, use_patchmatch=True)
inpainted_pm = inpainted_pm * 255
inpainted_pm = inpainted_pm.astype(np.uint8)
patch_orig = img_orig[y0:y1, x0:x1, :]
fft_orig = np.fft.fft2(rgb2gray(patch_orig))
fft_orig = np.log(np.abs(np.fft.fftshift(fft_orig)))
patch_inpaint = inpainted[y0:y1, x0:x1, :]
fft_inpaint = np.fft.fft2(rgb2gray(patch_inpaint))
fft_inpaint = np.log(np.abs(np.fft.fftshift(fft_inpaint)))
patch_inpaint_pm = inpainted_pm[y0:y1, x0:x1, :]
fft_inpaint_pm = np.fft.fft2(rgb2gray(patch_inpaint_pm))
fft_inpaint_pm = np.log(np.abs(np.fft.fftshift(fft_inpaint_pm)))
plt.figure()
plt.subplot(331)
plt.imshow(img_orig)
plt.subplot(332)
plt.imshow(inpainted)
plt.subplot(333)
plt.imshow(inpainted_pm)
plt.subplot(334)
plt.imshow(patch_orig)
plt.subplot(335)
plt.imshow(patch_inpaint)
plt.subplot(336)
plt.imshow(patch_inpaint_pm)
plt.subplot(337)
plt.imshow(fft_orig, vmin=fft_orig.min(), vmax=fft_orig.max())
plt.colorbar()
plt.subplot(338)
plt.imshow(fft_inpaint, vmin=fft_orig.min(), vmax=fft_orig.max())
plt.colorbar()
plt.subplot(339)
plt.imshow(fft_inpaint_pm, vmin=fft_orig.min(), vmax=fft_orig.max())
plt.colorbar()
plt.show()

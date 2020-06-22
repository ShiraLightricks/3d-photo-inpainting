import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from thebrain.models.inpainting import Inpainting
from thebrain.vision.image_processing import resize_with_ratio
from skimage.color import rgb2gray
from networks import Inpaint_Color_Net
from scipy.ndimage.morphology import binary_dilation


def isimg(file):
    return os.path.splitext(file)[1][1:] in ('png', 'jpg', 'jpeg')


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


size = 2048
hole = 512

# Inpainting from The Brain
inpainter = Inpainting()
images_folder = '/Users/jchetboun/Results/Inpainting/imgs'
output_folder = '/Users/jchetboun/Results/Inpainting/imgs_for_inpainting_circular'
os.makedirs(output_folder, exist_ok=True)
image_files = [file for file in os.listdir(images_folder) if isimg(file)]

# Inpainting from 3DPI
rgb_model = Inpaint_Color_Net()
rgb_feat_weight = torch.load('/Users/jchetboun/Projects/3d-photo-inpainting/checkpoints/color-model.pth', map_location=torch.device('cpu'))
rgb_model.load_state_dict(rgb_feat_weight)
rgb_model.eval()
rgb_model = rgb_model.to('cpu')

for file in image_files:
    print('Processing', file)
    # Image
    img_orig = imageio.imread(os.path.join(images_folder, file))[..., :3]
    img_orig = resize_with_ratio(img_orig, (size, size))
    x0 = (img_orig.shape[0] - hole) // 2
    y0 = (img_orig.shape[1] - hole) // 2
    # Mask
    # msk_for_inpaint = np.zeros(img_for_inpaint.shape[:2], dtype=np.uint8)
    # msk_for_inpaint[x0:x0+hole, y0:y0+hole] = 255
    msk_for_inpaint = create_circular_mask(img_orig.shape[0], img_orig.shape[1], center=None, radius=256)
    msk_for_inpaint = msk_for_inpaint.astype(np.uint8) * 255
    imageio.imwrite(os.path.join(output_folder, os.path.splitext(file)[0] + '_mask.png'), msk_for_inpaint)
    img_for_inpaint = np.copy(img_orig)
    img_for_inpaint[msk_for_inpaint > 0] = np.array([0, 0, 0], dtype=np.uint8)
    imageio.imwrite(os.path.join(output_folder, os.path.splitext(file)[0] + '.png'), img_for_inpaint)
    # Inpainting with The Brain
    inpainted = inpainter(image=img_for_inpaint, mask=msk_for_inpaint, use_patchmatch=False)
    inpainted = inpainted * 255
    inpainted = inpainted.astype(np.uint8)
    imageio.imwrite(os.path.join(output_folder, os.path.splitext(file)[0] + '_inpainted.png'), inpainted)
    # Inpainting with The Brain and dilated mask
    # msk_dilated = binary_dilation(msk_for_inpaint, iterations=5)
    # msk_dilated = msk_dilated.astype(np.uint8) * 255
    # inpainted_dilated = inpainter(image=img_for_inpaint, mask=msk_dilated, use_patchmatch=False)
    # inpainted_dilated = inpainted_dilated * 255
    # inpainted_dilated = inpainted_dilated.astype(np.uint8)
    # imageio.imwrite(os.path.join(output_folder, os.path.splitext(file)[0] + '_inpainted_dilated.png'), inpainted_dilated)
    # # inpainted_pm = inpainter(image=img_for_inpaint, mask=msk_for_inpaint, use_patchmatch=True)
    # # inpainted_pm = inpainted_pm * 255
    # # inpainted_pm = inpainted_pm.astype(np.uint8)
    # # imageio.imwrite(os.path.join(output_folder, os.path.splitext(file)[0] + '_inpainted_pm.png'), inpainted_pm)
    # # Inpainting with 3DPI
    # t_mask = np.zeros(img_orig.shape[:2])
    # t_mask[msk_for_inpaint > 0] = 1.
    # t_mask = torch.FloatTensor(t_mask).to('cpu')[None, None, ...]
    # t_context = np.zeros(img_orig.shape[:2])
    # t_context[msk_for_inpaint == 0] = 1.
    # t_context = torch.FloatTensor(t_context).to('cpu')[None, None, ...]
    # t_rgb = img_for_inpaint.astype(np.float) / 255.
    # t_rgb = torch.FloatTensor(t_rgb.transpose(2, 0, 1)).to('cpu')[None, ...]
    # t_edge = torch.zeros_like(t_mask)
    # inpainted_3dpi = rgb_model.forward_3P(t_mask, t_context, t_rgb, t_edge, unit_length=128, cuda='cpu', mode='orig')
    # inpainted_3dpi = inpainted_3dpi[0].numpy()
    # inpainted_3dpi = np.swapaxes(inpainted_3dpi, 0, 2)
    # inpainted_3dpi = np.swapaxes(inpainted_3dpi, 0, 1)
    # inpainted_3dpi = inpainted_3dpi * 255
    # inpainted_3dpi = inpainted_3dpi.astype(np.uint8)
    # imageio.imwrite(os.path.join(output_folder, os.path.splitext(file)[0] + '_inpainted_3dpi.png'), inpainted_3dpi)
    # # FFT analysis
    # patch_orig = img_orig[x0:x0+hole, y0:y0+hole, :]
    # fft_orig = np.fft.fft2(rgb2gray(patch_orig))
    # fft_orig = np.log(np.abs(np.fft.fftshift(fft_orig)))
    # patch_inpaint = inpainted[x0:x0+hole, y0:y0+hole, :]
    # fft_inpaint = np.fft.fft2(rgb2gray(patch_inpaint))
    # fft_inpaint = np.log(np.abs(np.fft.fftshift(fft_inpaint)))
    # # patch_inpaint_pm = inpainted_pm[x0:x0+hole, y0:y0+hole, :]
    # # fft_inpaint_pm = np.fft.fft2(rgb2gray(patch_inpaint_pm))
    # # fft_inpaint_pm = np.log(np.abs(np.fft.fftshift(fft_inpaint_pm)))
    # patch_inpaint_3dpi = inpainted_3dpi[x0:x0 + hole, y0:y0 + hole, :]
    # fft_inpaint_3dpi = np.fft.fft2(rgb2gray(patch_inpaint_3dpi))
    # fft_inpaint_3dpi = np.log(np.abs(np.fft.fftshift(fft_inpaint_3dpi)))
    # plt.figure(figsize=(12.8, 9.6))
    # plt.subplot(331)
    # plt.imshow(img_orig)
    # plt.axis('off')
    # plt.subplot(332)
    # plt.imshow(inpainted)
    # plt.axis('off')
    # plt.subplot(333)
    # # plt.imshow(inpainted_pm)
    # plt.imshow(inpainted_3dpi)
    # plt.axis('off')
    # plt.subplot(334)
    # plt.imshow(patch_orig)
    # plt.axis('off')
    # plt.subplot(335)
    # plt.imshow(patch_inpaint)
    # plt.axis('off')
    # plt.subplot(336)
    # # plt.imshow(patch_inpaint_pm)
    # plt.imshow(patch_inpaint_3dpi)
    # plt.axis('off')
    # plt.subplot(337)
    # plt.imshow(fft_orig, vmin=fft_orig.min(), vmax=fft_orig.max())
    # plt.axis('off')
    # plt.colorbar()
    # plt.subplot(338)
    # plt.imshow(fft_inpaint, vmin=fft_orig.min(), vmax=fft_orig.max())
    # plt.axis('off')
    # plt.colorbar()
    # plt.subplot(339)
    # # plt.imshow(fft_inpaint_pm, vmin=fft_orig.min(), vmax=fft_orig.max())
    # plt.imshow(fft_inpaint_3dpi, vmin=fft_orig.min(), vmax=fft_orig.max())
    # plt.axis('off')
    # plt.colorbar()
    # plt.savefig(os.path.join(output_folder, os.path.splitext(file)[0] + '_fft.png'))

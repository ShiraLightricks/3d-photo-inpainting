import os
import imageio as io

source_folder = "/Users/jchetboun/Projects/3d-photo-inpainting/image/from_lior"
inpainting_folder = "/Users/jchetboun/Projects/3d-photo-inpainting/save_lior"

all_source = os.listdir(source_folder)
all_inpainting = os.listdir(inpainting_folder)

for file_source in all_source:
    num = 0
    mean_width = 0
    mean_height = 0
    for file_inpainting in all_inpainting:
        if os.path.splitext(file_source)[0] in os.path.splitext(file_inpainting)[0] and "_img" in os.path.splitext(file_inpainting)[0]:
            print("Reading", file_inpainting)
            img = io.imread(os.path.join(inpainting_folder, file_inpainting))
            num += 1
            mean_width += img.shape[1]
            mean_height += img.shape[0]
    if num != 0:
        mean_width /= num
        mean_height /= num
    print("Mean size for", file_source, num, mean_height, mean_width)
    print("\n")

num = 0
mean_width = 0
mean_height = 0
for file_inpainting in all_inpainting:
    if "_img" in os.path.splitext(file_inpainting)[0]:
        print("Reading", file_inpainting)
        img = io.imread(os.path.join(inpainting_folder, file_inpainting))
        num += 1
        mean_width += img.shape[1]
        mean_height += img.shape[0]
mean_width /= num
mean_height /= num
print("Mean size for all", num, mean_height, mean_width)
print("\n")

import glob
import numpy as np
import os
from tqdm import tqdm

i3d_directory = "/media/valter/experiments/datasets/ucf101/i3d_1024_25fps_stack24step24_2stream_npy"
vggish_directory = "/media/valter/experiments/datasets/ucf101/vggish_npy"

i3d_files = glob.glob(i3d_directory+"/*_rgb.npy")
vggish_files = glob.glob(vggish_directory+"/*.npy")
i3d_fnames = []

for file in i3d_files:
    file = file.split("/")[-1]
    file = file.replace("_rgb.npy","")
    i3d_fnames.append(file)

for file in tqdm(i3d_fnames):
    f_test_audio = os.path.join(vggish_directory,file+"_vggish.npy")
    f_test_rgb = os.path.join(i3d_directory, file+"_rgb.npy")
    if not os.path.isfile(f_test_audio):
        x = np.load(f_test_rgb)
        y = np.zeros((x.shape[0],128))        
        np.save(f_test_audio, y)
import PIL
from moviepy import editor
import os
import numpy as np
from time import time

fps = 8

src_ls = [
    "./TRAIN_data_224/Bosch_GluePen_Akku-Heißklebepistole/",
    "./TEST_data_224/Bosch_GluePen_Akku-Heißklebepistole/",
    "./TRAIN_data_224/Bosch_PBH_2800_RE_Bohrhammer/",
    "./TEST_data_224/Bosch_PBH_2800_RE_Bohrhammer/",
    "./TRAIN_data_224/Bosch_PBH_3000-2_FRE_Bohrhammer/",
    "./TEST_data_224/Bosch_PBH_3000-2_FRE_Bohrhammer/",
    "./TRAIN_data_224/Bosch_PKP_18_E_Klebepistole/",
    "./TEST_data_224/Bosch_PKP_18_E_Klebepistole/",
    "./TRAIN_data_224/Bosch_PKP_3,6_LI_Akku-Heißklebepistole/",
    "./TEST_data_224/Bosch_PKP_3,6_LI_Akku-Heißklebepistole/",
    "./TRAIN_data_224/Bosch_PSB_14,4_LI-2_Akku-Zweigang-Schlagbohrschrauber_Lithium-Ionen/",
    "./TEST_data_224/Bosch_PSB_14,4_LI-2_Akku-Zweigang-Schlagbohrschrauber_Lithium-Ionen/",
    "./TRAIN_data_224/Bosch_PSB_18_LI-2_Akku-Zweigang-Schlagbohrschrauber_Lithium-Ionen/",
    "./TEST_data_224/Bosch_PSB_18_LI-2_Akku-Zweigang-Schlagbohrschrauber_Lithium-Ionen/",
    "./TRAIN_data_224/Bosch_PSB_500_RA_Schlagbohrmaschine/",
    "./TEST_data_224/Bosch_PSB_500_RA_Schlagbohrmaschine/",
    "./TRAIN_data_224/Bosch_PSB_570_RE_Schlagbohrmaschine/",
    "./TEST_data_224/Bosch_PSB_570_RE_Schlagbohrmaschine/",
    "./TRAIN_data_224/Bosch_PSB_850-2_RA_Schlagbohrmaschine/",
    "./TEST_data_224/Bosch_PSB_850-2_RA_Schlagbohrmaschine/",
]


def calc_duration_paths():
    num_frames = 0
    frames_paths = []
    for dir in src_ls:
        files = [os.path.join(dir, f) for f in os.listdir(dir)]
        num_frames += len(files)
        frames_paths.extend(files)
    return num_frames, frames_paths


num_frames, files = calc_duration_paths()

duration = num_frames // fps

time_list = list(np.arange(0, duration, 1. / fps))
img_dict = {a: f for a, f in zip(time_list, files)}


def make_frame(t):
    fpath = img_dict[t]
    im = PIL.Image.open(fpath).convert("RGB")
    ar = np.asarray(im)
    return ar


# gif_path = 'ani.gif'
clip = editor.VideoClip(make_frame, duration=duration)
# clip.write_gif(gif_path, fps=fps)
clip.write_videofile("{}.mp4".format(int(time())), fps=fps)

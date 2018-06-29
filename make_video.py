import PIL
from moviepy import editor
import os
import numpy as np
from time import time


SRC_DIR = "./TRAIN_data/Bosch_GluePen_Akku-Heißklebepistole/"
files = os.listdir(SRC_DIR)

fps = 16
duration = len(os.listdir(SRC_DIR)) // fps

time_list = list(np.arange(0, duration, 1. / fps))
img_dict = {a: f for a, f in zip(time_list, sorted(files))}


def make_frame(t):
    fpath = img_dict[t]
    im = PIL.Image.open(os.path.join(SRC_DIR, fpath))
    ar = np.asarray(im)
    return ar


# gif_path = 'ani.gif'
clip = editor.VideoClip(make_frame, duration=duration)
# clip.write_gif(gif_path, fps=fps)
clip.write_videofile("{}.mp4".format(int(time())), fps=fps)

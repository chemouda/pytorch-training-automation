import os
from PIL import Image
import numpy as np
from time import time
import multiprocessing as mp
from random import shuffle


TGT_SIZE = (224, 224)
RAW_DIR = "./RAW_data_v4"
TRAINSET_DIR = "./TRAIN_data_224_v5"
TESTSET_DIR = "./TEST_data_224_v5"


bgs = []
for bg in os.listdir("./bgs"):
    bgs.append(os.path.join("./bgs", bg))
shuffle(bgs)
bgs = bgs[:500]

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def merge_bg_fg(bg_idx, fg):
    bg = Image.open(bgs[bg_idx]).convert("RGBA")
    bg = bg.resize(TGT_SIZE)
    ret = Image.new("RGBA", bg.size)
    ret.paste(bg, (0, 0), bg)
    # Translation
    random_x = random_y = 0
    if TGT_SIZE[0] - fg.size[0] != 0 and TGT_SIZE[1] - fg.size[1] != 0:
        random_x = np.random.randint(0, TGT_SIZE[0] - fg.size[0])
        random_y = np.random.randint(0, TGT_SIZE[1] - fg.size[1])
    ret.paste(fg, (random_x, random_y), fg)
    return ret


def aux_gen_distortions(save_dir, file_path, timestamp, rand_int, scale, bg_idx):
    # load image again for forking can not pickle pil images
    # image load
    temp_im = Image.open(file_path)
    temp_im.thumbnail((scale, scale), Image.ANTIALIAS)
    rsz = temp_im.convert("RGBA")

    # for i in range(len(bgs)):
    final = merge_bg_fg(bg_idx=bg_idx, fg=rsz)
    file_name = "{}{}{}{}.png".format(rand_int, timestamp, int(time()), scale)
    file_path = os.path.join(save_dir, file_name)
    final.save(file_path)

    # FLIP
    flip_hor = rsz.transpose(method=Image.FLIP_LEFT_RIGHT)
    final = merge_bg_fg(bg_idx=bg_idx, fg=flip_hor)
    file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), 10, scale)
    file_path = os.path.join(save_dir, file_name)
    final.save(file_path)

    flip_ver = rsz.transpose(method=Image.FLIP_TOP_BOTTOM)
    final = merge_bg_fg(bg_idx=bg_idx, fg=flip_ver)
    file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), 20, scale)
    file_path = os.path.join(save_dir, file_name)
    final.save(file_path)

    flip_both = rsz.transpose(method=Image.TRANSPOSE)
    final = merge_bg_fg(bg_idx=bg_idx, fg=flip_both)
    file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), 30, scale)
    file_path = os.path.join(save_dir, file_name)
    final.save(file_path)

    # ROTATE
    for angle in [90, 180, 270]:
        rotated = rsz.rotate(angle)
        final = merge_bg_fg(bg_idx=bg_idx, fg=rotated)
        file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), angle, scale)
        file_path = os.path.join(save_dir, file_name)
        final.save(file_path)


def split_image(train_dir, test_dir, src, i):
    pool = mp.Pool(processes=64)
    res = []
    # image load
    im = Image.open(src)
    imgwidth, imgheight = im.size

    counter = i + 1

    view = im.crop((0, 0, imgwidth, imgheight))

    # NORMAL SCALE
    view.thumbnail(TGT_SIZE, Image.ANTIALIAS)
    rsz = view.convert("RGBA")

    # save image
    file_name = "{}{}{}{}{}.png".format(i, int(time()), i+1, 0, 1)
    file_path = os.path.join(train_dir, file_name)
    rsz.save(file_path)

    # remove white
    # datas = rsz.getdata()
    # new_data = []
    # for item in datas:
    #     if item[0] <= 70 and item[1] <= 70 and item[2] <= 70:
    #         new_data.append((255, 255, 255, 0))
    #     else:
    #         new_data.append(item)
    # rsz.putdata(new_data)

    for x in range(len(bgs)):
        print("{}/{}".format(x, len(bgs)))
        rand_int = np.random.randint(0, 1e6)
        timestamp = int(time())

        # NORMAL
        if counter % 2 == 0:
            res.append(pool.apply_async(aux_gen_distortions, args=(
                train_dir,
                src,
                timestamp,
                rand_int,
                TGT_SIZE[0],
                x,)))
        else:
            res.append(pool.apply_async(aux_gen_distortions, args=(
                test_dir,
                src,
                timestamp,
                rand_int,
                TGT_SIZE[0],
                x,)))

        if TGT_SIZE[0] < 64:
            return

        # SCALES
        for sz in [180, 190, 200]:
            if x % 2 == 0:
                res.append(pool.apply_async(aux_gen_distortions, args=(
                    train_dir,
                    src,
                    timestamp,
                    rand_int,
                    sz,
                    x,)))
            else:
                res.append(pool.apply_async(aux_gen_distortions, args=(
                    test_dir,
                    src,
                    timestamp,
                    rand_int,
                    sz,
                    x,)))
        counter += 1

    npl = len(res)
    npi = 1
    for r in res:
        print("{}/{}".format(npi, npl))
        r.get()
        npi += 1
    pool.close()


def read_panorama():
    dirs = os.listdir(RAW_DIR)
    for dir in dirs:
        path = os.path.join(RAW_DIR, dir)
        files = os.listdir(path)

        class_train_dir = os.path.join(TRAINSET_DIR, dir)
        class_test_dir = os.path.join(TESTSET_DIR, dir)
        create_dir_if_not_exists(class_train_dir)
        create_dir_if_not_exists(class_test_dir)

        for j in range(len(files)):
            print("{}::{}".format(j, dir))
            file_path = os.path.join(path, files[j])
            split_image(class_train_dir, class_test_dir, file_path, j)


def main():
    create_dir_if_not_exists(TRAINSET_DIR)
    create_dir_if_not_exists(TESTSET_DIR)
    read_panorama()


if __name__ == '__main__':
    main()

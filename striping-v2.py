import os
from PIL import Image
import numpy as np
from time import time


TGT_SIZE = (224, 224)
RAW_DIR = "./RAW_data"
TRAINSET_DIR = "./TRAIN_data_224_v2"
TESTSET_DIR = "./TEST_data_224_v2"


bgs = []
for bg in os.listdir("./bgs"):
    bgs.append(os.path.join("./bgs", bg))


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def merge_bg_fg(bg_idx_from, bg_idx_to, fg):
    i = np.random.randint(bg_idx_from, bg_idx_to)
    bg = Image.open(bgs[i]).convert("RGBA")
    bg = bg.resize(TGT_SIZE)
    ret = Image.new("RGBA", bg.size)
    ret.paste(bg, (0, 0), bg)
    # Translation
    random_x = random_y = 0
    if TGT_SIZE[0] - fg.size[0] != 0:
        random_x = np.random.randint(0, TGT_SIZE[0] - fg.size[0])
        random_y = np.random.randint(0, TGT_SIZE[1] - fg.size[1])
    ret.paste(fg, (random_x, random_y), fg)
    return ret


def aux_gen_distortions(save_dir, rsz, timestamp, rand_int, scale, bg_idx_from, bg_idx_to):
    # for i in range(len(bgs)):
    final = merge_bg_fg(bg_idx_from=bg_idx_from, bg_idx_to=bg_idx_to, fg=rsz)
    file_name = "{}{}{}{}.png".format(rand_int, timestamp, int(time()), scale)
    file_path = os.path.join(save_dir, file_name)
    final.save(file_path)

    # FLIP
    final = merge_bg_fg(bg_idx_from=bg_idx_from, bg_idx_to=bg_idx_to, fg=rsz)
    flip_hor = final.transpose(method=Image.FLIP_LEFT_RIGHT)
    file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), 10, scale)
    file_path = os.path.join(save_dir, file_name)
    flip_hor.save(file_path)

    final = merge_bg_fg(bg_idx_from=bg_idx_from, bg_idx_to=bg_idx_to, fg=rsz)
    flip_ver = final.transpose(method=Image.FLIP_TOP_BOTTOM)
    file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), 20, scale)
    file_path = os.path.join(save_dir, file_name)
    flip_ver.save(file_path)

    final = merge_bg_fg(bg_idx_from=bg_idx_from, bg_idx_to=bg_idx_to, fg=rsz)
    flip_both = final.transpose(method=Image.TRANSPOSE)
    file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), 30, scale)
    file_path = os.path.join(save_dir, file_name)
    flip_both.save(file_path)

    # ROTATE
    for angle in [90, 180, 270, 360]:
        final = merge_bg_fg(bg_idx_from=bg_idx_from, bg_idx_to=bg_idx_to, fg=rsz)
        rotated = final.rotate(angle)
        file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), angle, scale)
        file_path = os.path.join(save_dir, file_name)
        rotated.save(file_path)


def split_image(train_dir, test_dir, src):
    im = Image.open(src)
    imgwidth, imgheight = im.size
    step = imgheight // 18
    counter = 0
    for i in np.arange(0, imgheight, step):
        if i + step > imgheight:
            break
        view = im.crop((0, i, imgwidth, i + step))

        # NORMAL SCALE
        rsz = view.resize(TGT_SIZE).convert("RGBA")

        # save image
        file_name = "{}{}{}{}{}.png".format(i, int(time()), step, 0, 1)
        file_path = os.path.join(train_dir, file_name)
        rsz.save(file_path)

        # remove white
        datas = rsz.getdata()
        new_data = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        rsz.putdata(new_data)

        rand_int = np.random.randint(0, 1e6)
        timestamp = int(time())

        # NORMAL
        if counter % 2 == 0:
            aux_gen_distortions(save_dir=train_dir,
                                rsz=rsz,
                                timestamp=timestamp,
                                rand_int=rand_int,
                                scale=TGT_SIZE[0],
                                bg_idx_from=0,
                                bg_idx_to=25)
        else:
            aux_gen_distortions(save_dir=test_dir,
                                rsz=rsz,
                                timestamp=timestamp,
                                rand_int=rand_int,
                                scale=TGT_SIZE[0],
                                bg_idx_from=25,
                                bg_idx_to=50)

        if TGT_SIZE[0] < 64:
            return

        # SCALES
        for sz in [56, 112, 168]:
            sub_sz = rsz.resize((sz, sz))
            if counter % 2 == 0:
                aux_gen_distortions(save_dir=train_dir,
                                    rsz=sub_sz,
                                    timestamp=timestamp,
                                    rand_int=rand_int,
                                    scale=sz,
                                    bg_idx_from=0,
                                    bg_idx_to=25)
            else:
                aux_gen_distortions(save_dir=test_dir,
                                    rsz=sub_sz,
                                    timestamp=timestamp,
                                    rand_int=rand_int,
                                    scale=sz,
                                    bg_idx_from=25,
                                    bg_idx_to=50)
        counter += 1


def read_panorama():
    dirs = os.listdir(RAW_DIR)
    for dir in dirs:
        path = os.path.join(RAW_DIR, dir)
        files = os.listdir(path)
        file_path = os.path.join(path, files[0])
        class_train_dir = os.path.join(TRAINSET_DIR, dir)
        class_test_dir = os.path.join(TESTSET_DIR, dir)
        create_dir_if_not_exists(class_train_dir)
        create_dir_if_not_exists(class_test_dir)
        split_image(class_train_dir, class_test_dir, file_path)


def main():
    create_dir_if_not_exists(TRAINSET_DIR)
    create_dir_if_not_exists(TESTSET_DIR)
    read_panorama()


if __name__ == '__main__':
    main()

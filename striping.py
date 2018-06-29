import os
from PIL import Image
import numpy as np
from time import time


TGT_SIZE = (224, 224)
RAW_DIR = "./RAW_data"
TRAINSET_DIR = "./TRAIN_data_224"
TESTSET_DIR = "./TEST_data_224"


bgs = []
for bg in os.listdir("./bgs"):
    bgs.append(os.path.join("./bgs", bg))


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def aux_gen_distortions(train_dir, test_dir, rsz, timestamp, rand_int, scale):
    for i in range(len(bgs)):
        bg = Image.open(bgs[i]).convert("RGBA")
        bg = bg.resize(TGT_SIZE)
        # bg.paste(rsz, (0, 0), rsz)

        final = Image.new("RGBA", bg.size)
        # final = Image.alpha_composite(final, bg)
        # final = Image.alpha_composite(final, rsz)
        final.paste(bg, (0, 0), bg)
        final.paste(rsz, (0, 0), rsz)
        if rsz.size[0] == 64:
            final.show()
            return
        file_name = "{}{}{}{}.png".format(rand_int, timestamp, i, scale)
        if i < 18:
            file_path = os.path.join(train_dir, file_name)
        else:
            file_path = os.path.join(test_dir, file_name)
        final.save(file_path)

        # FLIP
        flip_hor = final.transpose(method=Image.FLIP_LEFT_RIGHT)
        file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, i, 10, scale)
        file_path = os.path.join(train_dir, file_name)
        flip_hor.save(file_path)

        flip_ver = final.transpose(method=Image.FLIP_TOP_BOTTOM)
        file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, i, 20, scale)
        file_path = os.path.join(test_dir, file_name)
        flip_ver.save(file_path)

        flip_both = final.transpose(method=Image.TRANSPOSE)
        file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, i, 30, scale)
        file_path = os.path.join(train_dir, file_name)
        flip_both.save(file_path)

        # ROTATE
        for angle in range(45, 361, 45):
            rotated = final.rotate(angle)
            file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, i, angle, scale)
            file_path = os.path.join(train_dir, file_name)
            if angle == 135 or angle == 315:
                file_path = os.path.join(test_dir, file_name)
            rotated.save(file_path)


def split_image(train_dir, test_dir, src):
    im = Image.open(src)
    imgwidth, imgheight = im.size
    step = imgheight // 18
    for i in np.arange(0, imgheight, step):
        if i + step > imgheight:
            break
        view = im.crop((0, i, imgwidth, i + step))

        # NORMAL SCALE
        rsz = view.resize(TGT_SIZE).convert("RGBA")

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
        aux_gen_distortions(train_dir=train_dir,
                            test_dir=test_dir,
                            rsz=rsz,
                            timestamp=timestamp,
                            rand_int=rand_int,
                            scale=TGT_SIZE[0])

        if TGT_SIZE[0] < 64:
            return

        # SCALES
        for sz in range(64, TGT_SIZE[0], 16):
            sub_sz = rsz.resize((sz, sz))
            aux_gen_distortions(train_dir=train_dir,
                                test_dir=test_dir,
                                rsz=sub_sz,
                                timestamp=timestamp,
                                rand_int=rand_int,
                                scale=sz)


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

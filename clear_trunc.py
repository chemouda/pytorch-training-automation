import os
from PIL import Image
import multiprocessing as mp

MP_CARDINALITY = 8


def aux_clear(sub_path):
    files = os.listdir(sub_path)
    for f in files:
        file_path = os.path.join(sub_path, f)
        try:
            Image.open(file_path).load()
        except:
            try:
                os.remove(file_path)
                print("Corrupt File @ {}".format(file_path))
            except:
                print("File does not exist @ {}".format(file_path))
                print("Corrupt File @ {}".format(file_path))


def clear_dir(dir):
    pool = mp.Pool(processes=MP_CARDINALITY)
    res = []

    dirs = os.listdir(dir)
    for d in dirs:
        sub_path = os.path.join(dir, d)
        print(sub_path)
        res.append(
            pool.apply_async(
                aux_clear,
                args=(
                    sub_path,
                )
            )
        )

    npl = len(res)
    npi = 1
    for r in res:
        print("{}/{}".format(npi, npl))
        r.get()
        npi += 1
    pool.close()


clear_dir("./TRAIN_data_224_v11")
clear_dir("./TEST_data_224_v11")
print("SUCCESSFUL TERMINATION")

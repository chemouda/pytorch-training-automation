import os


TRAINSET_DIR = "./TRAIN_data"
TESTSET_DIR = "./TEST_data"


def explore(dir):
    dirs = os.listdir(dir)

    for d in dirs:
        path = os.path.join(dir, d)
        print("{}, {}".format(len(os.listdir(path)), path))


explore(TRAINSET_DIR)
explore(TESTSET_DIR)
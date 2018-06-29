#!/usr/bin/python3

import pandas as pd
import os
import time
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
import urllib.request
from PIL import Image
import numpy as np
import cv2
import json as cPickle

TRAINSET_DIR = "./TRAIN_data"
TESTSET_DIR = "./TEST_data"


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


class CrawlerExtractor:
    def __init__(self, f_queries):
        if os.path.exists(f_queries):
            source_df = pd.read_csv(f_queries)
        else:
            raise Exception("CSV File Not Found")

        self.loop_fetch_products(source_df)

    def loop_fetch_products(self, df):
        # sample random set
        # df = df.sample(n=10, random_state=101)

        # create data directories
        if not os.path.exists(TRAINSET_DIR):
            os.mkdir(TRAINSET_DIR)

        if not os.path.exists(TESTSET_DIR):
            os.mkdir(TESTSET_DIR)

        self.bgs = []
        for bg in os.listdir("./bgs"):
            self.bgs.append(os.path.join("./bgs", bg))

        # loop
        for i, row in df.iterrows():
            print("\rprocess\t%d \r" % i)
            ean = row["EAN"]
            url = row["url"]

            print(ean, url)

            class_train_dir = os.path.join(TRAINSET_DIR, str(ean))
            class_test_dir = os.path.join(TESTSET_DIR, str(ean))

            if not os.path.exists(class_train_dir):
                os.mkdir(class_train_dir)

            if not os.path.exists(class_test_dir):
                os.mkdir(class_test_dir)

            # try:
            is_3d = self.scrap_bosch(class_train_dir, class_test_dir, url, ean)
            print(is_3d)
            # except:
            #     print("Skip: {}".format(ean))

    def scrap_bosch(self, train_dir, test_dir, url, ean):
        page = requests.get(url)
        if page.status_code != 200:
            return None
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.find("div", id="pbild_buehne")
        iter = results.children
        for c in iter:
            if isinstance(c, Tag):
                src = c["src"]
                self.store_train_image(train_dir, test_dir, src, ean)

        result = soup.find("img", id="PanoramaPicture")
        if result:
            print("++++++ {} ".format(result["src"]))

        # gallery images
        results = soup.find("div", class_="gal_images")
        iter = results.children
        for c in iter:
            if isinstance(c, Tag):
                print(c)
                src = c["src"]
                self.store_train_image(train_dir, test_dir, src, ean)

        results = soup.find("div", class_="gal_images")
        iter = results.children
        found = False
        for c in iter:
            if isinstance(c, Tag):
                if c["id"] == "PanoramaPicture":
                    found = True
                    src = c["src"]
                    tgt = self.store_temp_image(train_dir, test_dir, src, ean)
                    self.split_image(train_dir, test_dir, src, tgt, ean)
        return found

    def store_train_image(self, train_dir, test_dir, src, ean):
        file_name = "{}{}.png".format(np.random.randint(0, 1e6), int(time.time()))
        file_path = os.path.join(train_dir, file_name)
        t = urllib.request.urlretrieve(src, file_path)
        return file_path

    def store_temp_image(self, train_dir, test_dir, src, ean):
        file_name = "temp.jpg"
        t = urllib.request.urlretrieve(src, file_name)
        return file_name

    def split_image(self, train_dir, test_dir, src, tgt, ean):
        im = Image.open(tgt)
        imgwidth, imgheight = im.size
        step = imgheight // 18
        for i in np.arange(0, imgheight, step):
            if i + step > imgheight:
                break
            view = im.crop((0, i, imgwidth, i + step))
            rsz = im.resize((32, 32))

            for i in range(len(self.bgs)):
                bg = Image.open(self.bgs[i])
                bg = bg.resize((32, 32))
                # bg.paste(rsz, (0, 0), rsz)

                final = Image.new("RGBA", bg.size)
                final = Image.alpha_composite(final, bg)
                final = Image.alpha_composite(final, rsz)

                file_name = "{}{}{}.png".format(np.random.randint(0, 1e6), int(time.time()), i)
                if i < 2:
                    file_path = os.path.join(train_dir, file_name)
                else:
                    file_path = os.path.join(test_dir, file_name)
                print("360: {}".format(file_path))
                final.save(file_path)


def main():
    create_dir_if_not_exists(TRAINSET_DIR)
    create_dir_if_not_exists(TESTSET_DIR)
    CrawlerExtractor("mini_set.csv")


if __name__ == '__main__':
    main()

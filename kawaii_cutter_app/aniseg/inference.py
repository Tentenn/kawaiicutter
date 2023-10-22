import os
import sys
import argparse
import shutil

import cv2
import torch
import numpy as np
import glob
from torch.cuda import amp
from tqdm import tqdm
from PIL import Image, ImageOps
from django.conf import settings

from .train import AnimeSegmentation, net_names

from datetime import datetime

def add_log(log_text:str):
    # Get the current date and time
    now = datetime.now()

    # Format it as per your requirement
    formatted_date = now.strftime('%H:%M:%S_%d-%m-%y')

    with open("logs.txt", "a") as log:
        log.write(f"# {formatted_date}# {log_text}\n")

def get_mask(model, input_img, use_amp=True, s=640):
    h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
    ph, pw = s - h, s - w
    tmpImg = np.zeros([s, s, 3], dtype=np.float32)
    tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred.cpu().numpy().transpose((1, 2, 0)), (w0, h0))[:, :, np.newaxis]
        return pred

def append_to_db(line, db_path):
    with open(db_path, "a") as w:
        w.write(line+"\n")

def read_last_db_entry_num(db_path):
    with open(db_path, "r") as db:
        last_line = db.readlines()[-1]
    return int(last_line.replace('"', '').split(";")[0][1:])

def cv2_grayscale_to_rgba_pil(cv2_grayscale_image):
    # Convert the OpenCV grayscale image to a PIL grayscale image
    bw_pil_image = Image.fromarray(cv2_grayscale_image)

    # Create an RGBA image with the same size as the grayscale image
    rgba_image = Image.new("RGBA", bw_pil_image.size)

    # Use the grayscale image as the alpha channel for the RGBA image
    rgba_image.putalpha(bw_pil_image - 255)

    return rgba_image

class Segmentor():
    def __init__(self, ckpt, device="cpu"):
        add_log("...initializing model...")
        self.net = "isnet_is"
        self.ckpt = ckpt
        self.device = device
        self.model = AnimeSegmentation.try_load(self.net, self.ckpt, self.device, img_size=1024)
        add_log("Model initialized")

    def segment(self, path, file_name):
        add_log("segmenting...")
        # forward pass through model
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = get_mask(self.model, img, use_amp=not True, s=1024)
        add_log("got mask")
        img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        tmp_name = "temporary_file.png"
        cv2.imwrite("temporary_file.png", mask * 255)
        mask = ImageOps.invert(Image.open("temporary_file.png"))
        # file_name = "masked.png"

        # Save the image to the static directory
        image_path = os.path.join(settings.BASE_DIR, 'static', 'generated_images', file_name)
        mask.save(image_path, "PNG")
        add_log("image saved")
        return img

    def compare_models(test_set=""):
        pass

def main1():
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--net', type=str, default='isnet_is',
                        choices=net_names,
                        help='net name')
    parser.add_argument('--ckpt', type=str, default='saved_models/epoch=31,f1=0.9677.ckpt',
                        help='model checkpoint path')
    parser.add_argument('--data', type=str, default='in',
                        help='input data dir')
    parser.add_argument('--out', type=str, default='out',
                        help='output dir')
    parser.add_argument('--db_path', type=str, default='',
                        help='database file')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='hyperparameter, input image size of the net')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda:0')
    parser.add_argument('--fp32', action='store_true', default=True,
                        help='disable mix precision')
    parser.add_argument('--only-matted', action='store_true', default=False,
                        help='only output matted image')
    parser.add_argument('--add-db', action='store_true', default=False,
                        help='Add the card number to the database?')

    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device)

    model = AnimeSegmentation.try_load(opt.net, opt.ckpt, opt.device, img_size=opt.img_size)
    model.eval()
    model.to(device)

    if not os.path.exists(opt.out):
        os.mkdir(opt.out)

    for i, path in enumerate(tqdm(sorted(glob.glob(f"{opt.data}/*.*")))):
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = get_mask(model, img, use_amp=not opt.fp32, s=opt.img_size)
        if opt.only_matted:
            if opt.add_db:
                last_entry = read_last_db_entry_num(opt.db_path)
                tid = f"T{last_entry+1:05d}"
                print(tid, last_entry, path)

                # cv2.imwrite(f'{opt.out}/{tid}C.png', img)
                shutil.copy(path, f'{opt.out}/{tid}C.png')
                img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

                cv2.imwrite(f'{opt.out}/{tid}B.png', img)
                tmp_name = "temporary_file.png"
                cv2.imwrite("temporary_file.png", mask * 255)
                mask = ImageOps.invert(Image.open("temporary_file.png"))
                mask.save(f'{opt.out}/{tid}M.png')

                append_to_db(f"{tid};;;{path};;;", opt.db_path)

                '''imgmask = mask * 255
                for t in [0.5,0.55, 0.6,0.65, 0.7, 0.9]:
                    imgmask[imgmask < t*255] = 0
                    cv2.imwrite(f'{opt.out}/{i:06d}-{t}-M.png', imgmask)'''
            else:
                save_name = path.split("/")[-1].split("\\")[-1].replace("C","").replace(".png", "")
                img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(f'{opt.out}/{save_name}B.png', img)
                tmp_name = "temporary_file.png"
                cv2.imwrite("temporary_file.png", mask * 255)
                mask = ImageOps.invert(Image.open("temporary_file.png"))
                mask.save(f'{opt.out}/{save_name}M.png')
        else:
            imgmask = mask.copy()
            imgmask[imgmask < 0.65] = 0
            # mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            # imgmask = np.repeat(imgmask[:, :, np.newaxis], 3, axis=2)
            img = np.concatenate((img, mask * img, mask.repeat(3, 2) * 255, imgmask * img, imgmask.repeat(3, 2) * 255), axis=1).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{opt.out}/{i:06d}.jpg', img)


def main():
    # test Segmentor class
    seg = Segmentor("saved_models/epoch=31,f1=0.9677.ckpt")
    seg.segment("data/test.png")


if __name__ == "__main__":
    main()

    
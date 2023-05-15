import cv2
import numpy as np

# num = 40
# root = "./results/fwa/"

num = 20
root = "./results/bbfwa/"

with open(f"{root}/min_value.txt", "r") as f:
    content = f.read().split("\n")[:num]
    res = [float(item.split(": ")[1]) for item in content]

gif_list= []
for i in range(num):

    iter = i * 10

    img0 = cv2.resize(cv2.imread(f"{root}/0_degree/iter_{iter}.png"), (1500, 1500))
    img1 = cv2.resize(cv2.imread(f"{root}/50_degree/iter_{iter}.png"), (1500, 1500))
    imgs = np.concatenate([img0, img1], axis=1)

    text_img = np.ones((400, imgs.shape[1], 3), dtype=imgs.dtype) * 255
    text_img = cv2.putText(text_img, f"iter {iter}: {res[i]:.6f}", (1250, 200), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), thickness=6)

    show_img = np.concatenate([text_img, imgs], axis=0)

    h, w= show_img.shape[:2]
    wd = 800
    hd = int(h / w * wd)
    show_img = cv2.cvtColor(cv2.resize(show_img, (wd, hd)), cv2.COLOR_BGR2RGB)

    gif_list.append(show_img)

import imageio

imageio.mimsave(f"{root}/res.gif", gif_list, 'GIF', duration=0.5)

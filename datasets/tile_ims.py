import cv2
import numpy as np

def tile_im(im, name='1'):
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    y1 = 0
    M = 512
    N = 512

    for x in range(0,imgwidth,M):
        for y in range(0, imgheight, N):
            x1 = x + M
            y1 = y + N
            tiles = im[x:x+M,y:y+N, :]
            zeros = np.ones((M, N, 3)) * 255
            zeros2 = zeros.copy()
            zeros[:tiles.shape[0], :tiles.shape[1], :] = tiles
            zeros = np.concatenate([zeros, zeros2], 1)
            cv2.rectangle(im, (x, y), (x1, y1), (0, 255, 0))
            cv2.imwrite(f"save/{name}{str(x)}_{str(y)}.png", zeros)
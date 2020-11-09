import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import json
import cv2
import os
import argparse
from multiprocessing import Pool

def get_xy(tile, api_key):
    tile_search = 'https://avoin-paikkatieto.maanmittauslaitos.fi/geocoding/v1/pelias/' + \
                  'search?sources=mapsheets-tm35&lang=fin&crs=http://www.opengis.net/def/crs/EPSG/0/' + \
                  '3067&request-crs=http://www.opengis.net/def/crs/EPSG/0/3067&size=10&text='
    res = requests.get(tile_search + tile, auth=HTTPBasicAuth(api_key, ''))
    orig = json.loads(res.text)['features'][0]['geometry']['coordinates'][0]
    return orig[0][0], orig[0][1]

def render_contours(img, x, y, d, api_key, feature='tieviiva', color = (0, 0, 0)):
    features = 'https://avoin-paikkatieto.maanmittauslaitos.fi/maastotiedot/features/v1/'
    query = f'collections/{feature}/items?bbox={x},{y},{x + d},{y + d}'
    end = '&bbox-crs=http://www.opengis.net/def/crs/EPSG/0/3067&crs=http://www.opengis.net/def/crs/EPSG/0/3067'
    res = requests.get(features + query + end, auth=HTTPBasicAuth(api_key, ''))
    feats = json.loads(res.text)['features']
    if len(feats) != 0:
        contours = [feat['geometry']['coordinates'] for feat in feats]
    else:
        contours = np.array([])

    if feature not in ['korkeuskayra', 'tieviiva']:
        contours = [[cont] if len(cont)==1 else cont for cont in contours]
        contours = [np.array(item) for it in contours for item in it]

        for contour in contours:
            if contour.shape[0] == 1:
                sel = np.random.choice(len(COL_DICT.keys()) - 2) + 2
                sel = list(COL_DICT.keys())[sel]
                cont = contour
                cont = np.round(cont - [[x, y]]).astype(np.int32)
                img = cv2.fillPoly(img, cont, color=COL_DICT[sel])

    else:
        contours = [np.array(cont) - [[x, y]] for cont in contours]
        contours = [np.round(cont).astype(np.int32) for cont in contours]
        img = cv2.polylines(img, contours, isClosed=False, color=color, thickness=3)

    return img

def get_image(x, y, d, key, target='korkeuskayra'):

    img = np.ones((d, d, 3), np.uint8) * 255
    for sym in SYMBOLS:
        img = render_contours(img, x, y, d, key, sym)
    img = render_contours(img, x, y, d, key, 'korkeuskayra', COL_DICT['Brown'])
    img = render_contours(img, x, y, d, key)
    img2 = np.ones((d, d, 3), np.uint8) * 255
    img2 = render_contours(img2, x, y, d, key, 'korkeuskayra')
    img = np.concatenate([img[::-1], img2[::-1]], axis=1)
    return img

def image_write(tile, offset_x, offset_y, d, path_AB, api_key):

    x_t, y_t = get_xy(tile, api_key)
    x_t = x_t + offset_x
    y_t = y_t + offset_y
    img = get_image(x_t, y_t, d, api_key, 'korkeuskayra')
    img = cv2.resize(img, (2 * args.dimension, args.dimension))
    cv2.imwrite(path_AB, img)

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--directory', dest='directory', help='output directory', type=str, default='../mydataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='total number of images', type=int, default=1000)
parser.add_argument('--ratio', dest='ratio', help='portion of test', type=float, default=0.1)
parser.add_argument('--dimension', dest='dimension', help='dimension of images', type=int, default=512)
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true',default=False)
args = parser.parse_args()

COL_DICT = {'Brown': (0.0, 92.004, 209.1),
            'Black': (0., 0., 0.),
            'Blue': (255.0, 255., 0),
            'Green50yellow100': (0.0, 186.15, 158.1),
            'Green100': (22.95, 255.0, 61.20),
            'Green50':(138.975, 255.0, 158.10),
            'Green': (208.59, 255.0, 216.24),
            'Yellow': (53.55, 186.15, 255.0),
            'Yellow50': (154.275, 220.575, 255.0),
            'Grey': (191.25, 191.25, 191.25),
            }

SYMBOLS = ['kallioalue', 'suo', 'hietikko', 'soistuma', 'niitty',
           'puisto', 'suojametsa', 'taytemaa','kivikko', 'maatalousmaa',
           'jarvi','matalikko']

blocks = np.loadtxt('datasets/blocks.txt', dtype=str)
with open('datasets/api_key.txt', 'r') as f:
    KEY = f.read()

splits = ['train', 'test']

if not args.no_multiprocessing:
    pool=Pool()

for sp in splits:
    args.ratio = 1 - args.ratio
    num_imgs = round(args.ratio * args.num_imgs)
    print('split = %s, create %d images' % (sp, num_imgs))
    img_fold_dir = os.path.join(args.directory, sp)
    if not os.path.isdir(img_fold_dir):
        os.makedirs(img_fold_dir)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        tile = np.random.choice(blocks)

        size = np.random.randint(0.8 * args.dimension, 2 * args.dimension)
        offset_x = np.random.randint(0, 48000 - args.dimension)
        offset_y = np.random.randint(0, 24000 - args.dimension)
        path_AB = os.path.join(img_fold_dir, str('%d.png' % n))
        if not args.no_multiprocessing:
            pool.apply_async(image_write, args=(tile, offset_x, offset_y,
                                                size, path_AB, KEY))

        else:
            img = get_image(args=(tile, offset_x, offset_y,
                                                size, path_AB, KEY))

if not args.no_multiprocessing:
    pool.close()
    pool.join()


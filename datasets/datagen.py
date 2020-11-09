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
    print(feature)
    if len(feats) != 0:
        contours = [np.array(feat['geometry']['coordinates']) for feat in feats]
    else:
        contours = np.array([])

    if feature not in ['korkeuskayra', 'tieviiva']:
        for contour in contours:
            if contour.shape[0] == 1:
                sel = np.random.choice(len(COL_DICT.keys()) - 2) + 2
                sel = list(COL_DICT.keys())[sel]
                cont = contour
                cont = np.round(cont - [[x, y]]).astype(np.int32)
                img = cv2.fillPoly(img, cont, color=COL_DICT[sel])

    else:
        contours = [cont - [[x, y]] for cont in contours]
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
    return np.concatenate([img[::-1], img2[::-1]], axis=1)

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--directory', dest='directory', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
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

with open('datasets/api_key.txt', 'r') as f:
    KEY = f.read()

DIM = 512
x, y = get_xy('M3133R', KEY)
img = get_image(x, y, 1000, KEY)
img = cv2.resize(img, (2*DIM, DIM))

winname = 'example'
cv2.namedWindow(winname)
cv2.imshow(winname, img)
cv2.waitKey()
cv2.destroyWindow(winname)

def get_input(d, offset_x, offset_y, im, tile, api_key):
    x_t, y_t = get_xy(tile, api_key)
    x_t = x_t + offset_x
    y_t = y_t + offset_y
    conts = get_contours(x_t, y_t, d, api_key)
    arr = get_array(conts, x_t, y_t, d)
    return im[-d - offset_y:-offset_y, offset_x:offset_x + d, :], arr

print(os.listdir('./dataset'))
"""
sp = ['train','test']
if not args.no_multiprocessing:
    pool=Pool()

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_list = os.listdir(img_fold_A)
    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_dir = os.path.join(args.directory, sp)
    if not os.path.isdir(img_fold_dir):
        os.makedirs(img_fold_dir)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_AB = os.path.join(img_fold_dir, name_AB)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_AB))
            else:
                im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)
if not args.no_multiprocessing:
    pool.close()
    pool.join()

"""
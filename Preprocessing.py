import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio as rio
from osgeo import gdal
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show
from rasterio.plot import plotting_extent
from shapely.geometry import Polygon, box
from shapely.geometry import mapping
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import math
path='C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA'
os.chdir(path)
croppath='C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA/Box.shp'
crop_extent=gpd.read_file(croppath)

rgbjp2s=['C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA/FaiziRed.tif','C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA/FaiziGreen.tif','C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA/FaiziBlue.tif']


croppedbands=[rgbjp2s[0],rgbjp2s[1],rgbjp2s[2]]


outputstackpath='C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA/stack.tif'

with rio.open(rgbjp2s[0]) as f:
	checkred=f.read(1, masked=True)
with rio.open(rgbjp2s[1]) as f:
	checkgreen=f.read(1, masked=True)
with rio.open(rgbjp2s[2]) as f:
	checkblue=f.read(1, masked=True)


croppedbands_stack, croppedbands_meta=es.stack(croppedbands, outputstackpath)
print(croppedbands_stack.shape)

imgred = cv2.normalize(checkred, 0, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
equred = cv2.equalizeHist(imgred)

imggreen = cv2.normalize(checkgreen, 0, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
equgreen = cv2.equalizeHist(imggreen)

imgblue = cv2.normalize(checkblue, 0, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
equblue = cv2.equalizeHist(imgblue)

img_out = cv2.merge((equblue, equgreen, equred))
print(img_out.shape)
plt.imshow(img_out)
cv2.imwrite('new.jpg', img_out)
plt.show()


cropfordeploymentpath='C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA/Crop for deployment.shp'
cropfordeployment_extent=gpd.read_file(cropfordeploymentpath)

outputstackpath='C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA/stack.tif'

outputmaskpath='C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA/Unstruct_uncropped_masks2.tif'


outputmaskpath='C:/Users/Hadi Askari/Desktop/Sentinel 2/S2A_MSIL1C_20200301T054741_N0209_R048_T43SCT_20200301T085541.SAFE/GRANULE/L1C_T43SCT_A024498_20200301T055331/IMG_DATA/FaiziMask.tif'
with rio.open(outputmaskpath) as Ms:
    mask_start = Ms.read(1, masked=True)
    mask_start=(mask_start/255).astype('uint8')
    print(np.amax(mask_start))

def extract(img, window, slide=16, masked=True):
  (win_x, win_y) = window
  (h, w) = (img.shape[0], img.shape[1])
  extracted = []
  y_start = 0
  for y in range(math.floor(h/slide)):
    x_start = 0
    for x in range(math.floor(w/slide)):
        if not masked:
            arr = img[y_start: y_start + win_y, x_start: x_start + win_x, :]
        else:
            arr = img[y_start: y_start + win_y, x_start: x_start + win_x]
        extracted.append((arr))
        x_start += slide
    y_start += slide
  return extracted

def print_all(imgs, dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    for idx, img in enumerate(imgs):
        cv2.imwrite(f'{dir}/{idx}.png', img)
#

extracted = extract(img_out, (64,64))
print_all(extracted, 'Sentinal 2 Images for Fast AI')



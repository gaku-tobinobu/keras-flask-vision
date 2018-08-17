import math, operator
import numpy as np
from scipy import stats
import cv2
from keras.preprocessing.image import img_to_array
from collections import defaultdict


def convert2gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def blur_detection(gray):
    laplacian_var = cv2.Laplacian(gray, 5).var()
    blur_score = min(laplacian_var/3000, 1)
#     if laplacian_var<=300:
#         blur_score = 0
#     elif laplacian_var<=1000:
#         blur_score = 0.25
#     elif laplacian_var<=2000:
#         blur_score = 0.5
#     elif laplacian_var<=3000:
#         blur_score = 0.75
#     else:
#         blur_score = 1.00
    return round(blur_score,3)

def contrast_detection(gray):
    hist, _ = np.histogram(gray.flatten(),256,[0,256])
    nhist = hist/sum(hist)
    entropy = -sum([x*math.log2(x) for x in nhist if x>0])
    max_entropy = math.log2(256)
    norm_entropy = entropy/max_entropy
    contrast_score = round(1 - norm_entropy,3)
    return contrast_score

def image_sizing(image_pil, max_threshold =750000):
    _shp = image_pil.size
    _imagesize = np.prod(_shp)
    _size_score = _imagesize / max_threshold
    _size_score = round(min(_size_score, 1),3)
    return _size_score

def average_pixel_width(image):  
    im_array = np.asarray(image.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (image.size[0]*im.size[1]))
    return apw

def color_analysis(image):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in image.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count), 2)
    dark_percent = round((float(dark_shade)/shade_count), 2)
    return light_percent, dark_percent

def perform_color_analysis(image, flag='max'):
    
    # cut the images into two halves as complete average may give bias results
    size = image.size
    halves = (size[0]/2, size[1]/2)
    im1 = image.crop((0, 0, size[0], halves[1]))
    im2 = image.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'max':
        return 1 - max(light_percent, dark_percent)
    elif flag == 'black':
        return 1 - dark_percent
    elif flag == 'white':
        return 1 - light_percent
    else:
        return None

def image_quality_detection(image, mode='score'):
    size_score = image_sizing(image)
    dullness_score = perform_color_analysis(image)
#     pix_width_score = average_pixel_width(image)
    image = img_to_array(image)                          
    gray = convert2gray(image)
    blur_score = blur_detection(gray)
    contrast_score = contrast_detection(gray)
    quality_score = np.mean([blur_score, contrast_score, size_score, dullness_score])
    if mode=='score':
        return round(quality_score,3)
    elif mode == 'all':
        score_dict = {'score': round(quality_score,3), 'blur':blur_score, 'contrast': contrast_score,\
         'size': size_score, 'dull': dullness_score}
        return score_dict
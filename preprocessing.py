import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math

from scipy.interpolate import PPoly, splder, splev, splrep

# Automatic brightness and contrast optimization with optional histogram clipping
# works on greyScale image
def automatic_brightness_and_contrast(grey_image, clip_hist_percent=1):
    print("Running automatic_brightness_and_contrast with clip_hist_percent={}".format(clip_hist_percent))

    # Calculate grayscale histogram
    hist = cv2.calcHist([grey_image],[0],None,[256],[0,256])
    hist_size = len(hist)
    # print(hist_size)
    
    # Calculate cumulative distribution of histogram 
    # to determine where pixel intensity is less than some threshold
    # the default threshold is 1%
    hist_acc = []
    hist_acc.append(float(hist[0]))
    for idx in range(1, hist_size):
        hist_acc.append(hist_acc[idx-1] + float(hist[idx]))

    # Calculate clipping threshold value
    max_acc_value = hist_acc[-1]
    clip_hist_percent *= max_acc_value/100.0
    clip_hist_percent /= 2.0

    # Calculate the index of left side for clipping
    min_grey = 0
    while hist_acc[min_grey] < clip_hist_percent:
        min_grey += 1

    # Calculate the index for right clipping range
    max_grey = hist_size - 1
    while hist_acc[max_grey] >= (max_acc_value - clip_hist_percent):
        max_grey -= 1


    # Calculate alpha and beta from the formula:
    # g(i,j) = alpha*f(i,j) + beta
    alpha = 255 / (max_grey - min_grey)
    beta = -min_grey * alpha

    enh_image = cv2.convertScaleAbs(grey_image, alpha=alpha, beta=beta)
    enh_hist = cv2.calcHist([enh_image], [0], None, [256], [0,256])

    # Plot of the histograms of the original and auto enhanced images
    # plt.suptitle("Greyscale histogram")
    # plt.plot(hist)
    # plt.plot(enh_hist)
    
    # plt.legend(["original image", "enhanced image"], loc ="best")
    # plt.xlim([0,256])
    # # plt.show()

    return enh_image, alpha, beta


# Calculating vertical and horizontal intensity projection
def intensity_projection(grey_image):

    ### Calculate vertical intensity projection
    sum_col = cv2.reduce(grey_image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    sum_col = sum_col.flat[:]

    ### Calculate horizontal intensity projection
    sum_row = cv2.reduce(grey_image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    sum_row = sum_row.flat[:]
    
    return sum_col, sum_row

    
# Calculating spine ROI ###
def detect_spine(image, sum_col, sum_row):
    image_width = image.shape[1]
    image_height = image.shape[0]

    ### Finding maximum peak of the vertical intensity projection
    max_col_value = np.max(sum_col)
    mean_col_value = np.mean(sum_col, dtype=int)
    
    # Finds columns around column with maximum intensity 
    # these are the columns with the head, spine and sacrum
    eps_col = mean_col_value//2
    # eps_col = 5*image_width//100
    # eps_col = 0
    col_max_values_idx = np.where(sum_col >= max_col_value - eps_col)
    np.append(col_max_values_idx[0], np.where(sum_col <= max_col_value + eps_col)[0] )

    # col_max_values_idx = np.where(sum_col >= mean_col_value - eps_col)
    # np.append(col_max_values_idx[0], np.where(sum_col <= mean_col_value + eps_col)[0] )

    col_max_values = np.zeros(sum_col.size, dtype=int)
    col_max_values[col_max_values_idx[0]] = sum_col[col_max_values_idx[0]]

    ### Finding minimum and maximum peaks of horizontal projection
    eps_row = 5*image_height//100
    min_max_row = np.zeros(sum_row.size, dtype=int)
    min_row_idx = np.argmin(sum_row)
    max_row_idx = np.argmax(sum_row[len(sum_row)//2:])
    max_row_idx += len(sum_row)//2

    # print("min_row_idx:", min_row_idx)
    # print("image_height//3", image_height//3)
    if min_row_idx > image_height//3: 
        min_row_idx = 0
        min_max_row[min_row_idx:min_row_idx+eps_row] = sum_row[min_row_idx:min_row_idx+eps_row]
    else:
        min_max_row[min_row_idx-eps_row:min_row_idx+eps_row] = sum_row[min_row_idx-eps_row:min_row_idx+eps_row]
    # print("min_row_idx:", min_row_idx)
    # min_max_row[min_row_idx-eps_row:min_row_idx+eps_row] = sum_row[min_row_idx-eps_row:min_row_idx+eps_row]

    if max_row_idx < 2*image_height//3:
        # max_row_idx = max_row_idx - eps_row
        max_row_idx = image_height - eps_row
    else:
        max_row_idx =  max_row_idx-2*eps_row
    min_max_row[max_row_idx-eps_row:max_row_idx+eps_row] = sum_row[max_row_idx-eps_row:max_row_idx+eps_row]

    
    # Finding starting index of vert_values of interest (values different from 0)
    idx_pairs = np.where(np.diff(np.hstack(([False], col_max_values != 0, [False]))))[0].reshape(-1,2)
    # print(idx_pairs)
    # Finding longest sequence of values different than 0
    longest_seq = idx_pairs[np.diff(idx_pairs).argmax()]
    # print(longest_seq[1])
    
    # spine_start = (max_col_idx-eps_v,min_row_idx-eps_row)
    # min_row_idx = np.argmin(sum_row)
    spine_start = (longest_seq[0], min_row_idx)
    spine_end = ( longest_seq[1], max_row_idx)
    # spine_end = ( longest_seq[1], image_height)
    # image_spine = cv2.rectangle(grey_image, spine_start,spine_end, (0,255,0), 5)
    
    # cv2.imshow("detected spine image", image_spine)
    return spine_start, spine_end, col_max_values, min_max_row
    

    # return image_spine


### Uses Fourier transformation to filter high frequency noise
def FF_denoising(image, r=8, hpf=0):
    print("Runing FF_denoising with r={} and hpf={} \n".format(r, hpf))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image)
    dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


    rows, cols = image.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    if hpf==1:
        mask[:] = 1
        mask[mask_area] = 0    
    else:
        mask[mask_area] = 1

    fshift = dft_shift * mask
    fshift_mask_mag = cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    
    ### reconstruct and normalize image values
    min, max =np.amin(image, (0,1)), np.amax(image, (0,1))
    min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
    img_back = cv2.normalize(img_back,None, alpha=0, beta=252, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
    
    plt.subplot(141),plt.imshow(image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(143),plt.imshow(fshift_mask_mag, cmap = 'gray')
    plt.title('Mask 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
    plt.title('image back'), plt.xticks([]), plt.yticks([])
    # cv2.imshow('image back norm', img_back)
    # plt.show()   
    
    return img_back 


# Contrast Limited Adaptive Histogram Equalization
def adaptive_equalization(image, clip_limit=2, tile_size=(16, 16)):
    print("Running adaptive_equalization with clip_limit={} and tile_size={}".format(clip_limit, tile_size))
    hist_eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    equalized_image = hist_eq.apply(image)    

    return equalized_image


def pyramid(image, layers_count):
    print("Generating image pyramid with {} layers".format(layers_count))
    
    layer = image.copy()

    # generating gaussian pyramid
    gp = [layer]
    for i in range(layers_count):
        layer = cv2.pyrDown(layer)
        gp.append(layer)
        # print("gaussian pyramid:{} - {} ".format(i, layer.shape))
    
    # # save gp layers to images
    # i = 0
    # for layer in gp:
    #     cv2.imwrite(os.path.join("results", "gp_"+str(i)+".png"), layer)
    #     i += 1
    

    kernal_size = 15
    # generating laplacian pyramid
    layer=gp[-1]
    layer_edge = cv2.Laplacian(gp[-1], cv2.CV_8U, ksize=kernal_size)
    layer1 = cv2.subtract(layer, layer_edge)

    lp = [layer_edge] 
    for i in range(layers_count, 0, -1):
        size = (gp[i-1].shape[1], gp[i-1].shape[0])
        gp_edge = cv2.Laplacian(gp[i-1], cv2.CV_8U, ksize=kernal_size)
        layer_edge = lp[-1]
        layer_edge = cv2.pyrUp(layer_edge,dstsize=size)
        laplacian = cv2.add(gp_edge, layer_edge)
        
        # layer_edge = cv2.convertScaleAbs(layer_edge)
        
        # laplacian = cv2.subtract(gp[i], layer_edge)
        # laplacian = cv2.pyrUp(laplacian,dstsize=size)
        # gaussian_extended = cv2.pyrUp(gp[i],dstsize=size)
        # gaussian_extended = cv2.Laplacian(gaussian_extended, cv2.CV_16S, ksize=3)
        # gaussian_extended = cv2.convertScaleAbs(gaussian_extended)
        # laplacian = cv2.add(gp[i-1], gaussian_extended)
        # laplacian = adaptive_equalization(laplacian, clip_limit=3, tile_size= (size[0]//4,size[1]//4))
        lp.append(laplacian)
        # print("laplacian pyramid:{} - {} ".format(i, laplacian.shape))
    
    # # saving lp layers to images
    # i = 0
    # for layer in lp:
    #     cv2.imwrite(os.path.join("results", "lp_"+str(i)+".png"), layer)
    #     i += 1


    # reconstructing image 
    reconstructed_image = cv2.subtract(gp[-1], lp[0])
    # cv2.imwrite(os.path.join("results", "reconstructed_0"+".png"), reconstructed_image)
    
    for i in range(1, layers_count+1):
        size = (lp[i].shape[1], lp[i].shape[0])
        # reconstructed_image = cv2.pyrUp(reconstructed_image,dstsize=size)
        # reconstructed_image = cv2.add(reconstructed_image, lp[i])
        reconstructed_image = cv2.subtract(gp[-(i+1)], lp[i])
        # cv2.imwrite(os.path.join("results", "reconstructed_"+str(i)+".png"), reconstructed_image)
    
    
    return reconstructed_image


def kmean(image, k=3):
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)


    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]


    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

import cv2
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pydicom as dicom
from scipy.ndimage import median_filter

from io import BytesIO
from pathlib import Path
import preprocessing as pr
import computation as compute
from computation import Point, Line
from scipy.ndimage import median_filter

def validate_is_dicom(file_content: bytes) -> bool:
    """Gets a byte type object and checks if the content from position 128 till 132 
    mathes the tag 'DICM' for DICOM files.

    Args:
        file_content (bytes): File to be validated

    Returns:
        bool: True if the content from 128 till 132 mathces 'DICM' tag
    """
    print("Validating if the file has a DICM tag ...")
    return file_content[128:132] == b'DICM'


def save_dicom(file_name: str, file_content: bytes) -> None:
    """Gets the uploaded byte type object and after anonymising it saves it as DICOM image with the given filename.

    Args:
        file_name (str): file name
        file_content (bytes): image content
    """
    print("Saving file content as DICOM file ...")
    dataset_to_write = dicom.dcmread(BytesIO(file_content))
    print(dataset_to_write.is_implicit_VR)
    
    # remove sensitive information in DICOM metadata as patient Name, ID, age, birth date and sex
    dataset_to_write = anonymise_dicom_data(dataset=dataset_to_write)
    
    # dicom.filewriter.write_file(file_name, file_content, False)
    file_name = Path(file_name)
    dicom.dcmwrite(filename=file_name, dataset=dataset_to_write, write_like_original=True)
    # TODO: change when pydicom version > 3.0.0
    # dicom.dcmwrite(filename=file_name, dataset=dataset, enforce_file_format=True)


def anonymise_dicom_data(dataset: dicom.FileDataset) ->  dicom.FileDataset:
    """Deletes sensitive data from DICOM type files to protect patient personal information.

    Args:
        dataset (dicom.FileDataset): Given DICOM image

    Returns:
        dicom.FileDataset: Same data set as the given one, but without Patient information as ID, Name, Sex, Age and birthdate
    """
    print("Anonymise DICOM data ...")
    dataset.PatientID = None
    dataset.PatientName = None
    dataset.PatientSex = None
    dataset.PatientAge = None
    dataset.PatientBirthDate = None

    if dataset.get("StudyDate", None):
        dataset.StudyDate = None
    if dataset.get("StudyTime", None):
        dataset.StudyTime = None
    if dataset.get("SeriesDate", None):
        dataset.SeriesDate = None
    if dataset.get("SeriesTime", None):
        dataset.SeriesTime = None

    return dataset


def read_all_images(image_directory):

    if not os.path.exists(image_directory):
        print("(>_<)  Oops! No such directory (-_-)  \n")
        return
    else: 
        print("reading image files in \"{}\" ...".format(image_directory))
        image_name = ''
        image_ext = 'png'
        images = []
        image_type = ''
        image_data = ''
    
    for filename in os.listdir(image_directory):
        filename_rel_path = os.path.join(image_directory,filename)
        if re.search("\\.|jpg|png|jepg|dcm",filename, re.I):
            name =  filename.split('.')
            image_name = name[0]
            image_ext = name[1]
            image_type = 'Color'
            image_data = cv2.imread(filename_rel_path)
            
        else:
            image_name = filename
            image_ext = 'png'
            ds = dicom.dcmread(filename_rel_path)
            image_type =  ds.PhotometricInterpretation  #usually dicom x-ray is MONOCHROME2
            if ds.pixel_array.any():
                ### convert byte raw image data into uint8 in range [0,255]
                image_data = ds.pixel_array - np.min(ds.pixel_array)
                image_data = image_data / np.max(image_data)
                image_data = (image_data * 255).astype(np.uint8)        
        
        if image_data.any():
            images.append((image_data, image_name, image_ext, image_type))

    return(images)


def open_image_file(filename, image_directory):
    # filename_rel_path = f"uploads\\{filename}"
    filename_rel_path = os.path.join(image_directory, filename)
    print(f"Opening image: {filename_rel_path}")

    if re.search("\\.|jpg|png|jepg|dcm",filename, re.I):
        name =  filename.split('.')
        image_name = name[0]
        image_ext = name[1]
        image_type = 'Color'
        image_data = cv2.imread(filename_rel_path)
    else:
        image_name = filename
        image_ext = 'png'
        ds = dicom.dcmread(filename_rel_path)
        image_type =  ds.PhotometricInterpretation        # usually dicom x-ray is MONOCHROME2

        if ds.pixel_array.any():
            if image_type == 'MONOCHROME1':
                px_data = 1 - ds.pixel_array
            elif image_type == 'MONOCHROME2':
                px_data = ds.pixel_array
            ### convert byte raw image data into uint8 in range [0,255]
            print("Converting byte raw data from dicom into uint8")
            image_data = px_data - np.min(px_data)
            image_data = image_data / np.max(image_data)
            image_data = (image_data * 255).astype(np.uint8)        

    if image_data.any():
        # print(f"image data: {image_data}")
        return (image_data, image_name, image_ext, image_type)
    else:
        print("Image data is missing")


def process_image(filename, image_directory, results_directory) -> list:
    result = []
    original_image, image_name, image_ext, image_type = open_image_file(filename, image_directory)
    
    print("Processing: {} - {} - {}".format(image_name, original_image.shape, image_type))
    
    if image_type != "MONOCHROME2":
        grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        grey_image = original_image
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    original_image_width = grey_image.shape[1]
    original_image_height = grey_image.shape[0]

    scale_ratio = 1 if original_image_height < 1500 else  0.2
    algorithms_strength = "weak" if original_image_height <= 1000 else "strong"
    print(f"Algorithm strength: {algorithms_strength}")

    image_height = int(original_image_height*scale_ratio)
    image_width = int(original_image_width*scale_ratio)

    grey_image = cv2.resize(grey_image, (image_width,image_height), interpolation = cv2.INTER_AREA)
    image = cv2.resize(original_image, (image_width,image_height)  , interpolation = cv2.INTER_AREA)
    print("Scale ratio: {} \nNew image size: {}-{}".format(scale_ratio, image_width, image_height))

    cv2.imwrite(os.path.join(results_directory,"{}_00-original_image.{}".format(str(image_name),str(image_ext))), image)
    
    ### auto image enhancement for improving brightness and contrast - histogram stretching
    clip_hist_percent = 7 if algorithms_strength == "strong" else 4
    enh_image, alpha, beta = pr.automatic_brightness_and_contrast(grey_image, clip_hist_percent=clip_hist_percent)
    cv2.imwrite(os.path.join(results_directory,"{}_01-enh-1_{}.{}".format(str(image_name),str(clip_hist_percent),str(image_ext))), enh_image)

    ### Denoising - apply strong bilateral filter
    # d = 33
    # sigma = 17
    # enh_image = cv2.bilateralFilter(enh_image, d, sigma, sigma)
    # cv2.imwrite(os.path.join(results_directory,"{}_02-bltr-{}-{}.{}".format(str(image_name),str(d),str(sigma),str(image_ext))), enh_image)

    ### Detecting spine ROI by horizontal projection
    sum_col, sum_row = pr.intensity_projection(enh_image)
    spine_start, spine_end, col_values, min_max_row = pr.detect_spine(enh_image, sum_col, sum_row, algorithms_strength)
    print(f"Cropped pos (cor, row): {spine_start}-{spine_end}")

    ### Plot intensity projection histograms and detected spine ROI
    fig = plt.figure()
    plt.suptitle("Intensity projection")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Vertical projection')
    ax2.title.set_text('Horizontal projection')

    plt.subplot(1, 2, 1)
    plt.bar(range(0,image_width), sum_col, align='edge', width=1.0, color='coral')
    plt.bar(range(0,image_width), col_values, align='edge', width=1.0, color='lightcoral')
    plt.axvline(x=spine_start[0], color='cyan')
    plt.axvline(x=spine_end[0], color='red')
    plt.ylabel('Intensity' )
    plt.xlabel('Image width')


    plt.subplot(1, 2, 2)
    plt.barh(np.arange(image_height), sum_row, align='center', height=1.0, color='turquoise')
    plt.barh(np.arange(image_height), min_max_row, align='center', height=1.0, color='paleturquoise')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.axhline(y=spine_start[1], color='yellow')
    plt.axhline(y=spine_end[1], color='red')
    plt.ylabel('Image height')
    plt.xlabel('Intensity')
    fig.tight_layout(pad=1.0)

    fig.set_figwidth(int(image_width*2.5)//100)
    fig.set_figheight(image_height//100)
    plt.savefig(os.path.join(results_directory,"{}_02-intensity_projection.{}".format(str(image_name),"png")))
    plt.clf()


    ### Cropping spine region
    spine_crop_enh = enh_image[spine_start[1]:spine_end[1], spine_start[0]:spine_end[0]]
    spine_crop = image[spine_start[1]:spine_end[1], spine_start[0]:spine_end[0]]
    spine_crop_grey = grey_image[spine_start[1]:spine_end[1], spine_start[0]:spine_end[0]]
    
    spine_height, spine_width = spine_crop_grey.shape
    print("Cropped spine size: {}".format(spine_crop_grey.shape))
    cv2.imwrite(os.path.join(results_directory,"{}_02-spine-crop.{}".format(str(image_name),str(image_ext))), spine_crop_grey)

    ### adaptive equalization for improving image contrast
    clip_limit = 1
    tile_size_per = 0.76
    tile_size = (spine_width//int(spine_width*tile_size_per),spine_height//int(spine_height*tile_size_per))
    spine_enh_image = pr.adaptive_equalization(spine_crop_enh, clip_limit=clip_limit, tile_size=tile_size)
    cv2.imwrite(os.path.join(results_directory,"{}_02-spine-enh_{}_{}.{}".format(str(image_name),str(clip_limit),str(tile_size),str(image_ext))), spine_enh_image)

    # ### auto image enhancement for improving brightness and contrast - histogram stretching
    # clip_hist_percent = 5 if algorithms_strength == "strong" else 2
    # spine_crop_enh, alpha, beta = pr.automatic_brightness_and_contrast(spine_crop_enh, clip_hist_percent=clip_hist_percent)
    # cv2.imwrite(os.path.join(results_directory,"{}_02-spine-enh_{}.{}".format(str(image_name),str(clip_hist_percent),str(image_ext))), spine_crop_enh)


    ### Denoising - apply strong bilateral filter
    d = 15
    sigma = 115
    # d = 1
    # sigma = 5    
    if algorithms_strength == "weak":
        d = 1
        sigma = 5
    spine_crop_blt = cv2.bilateralFilter(spine_crop_enh, d, sigma, sigma)
    cv2.imwrite(os.path.join(results_directory,"{}_03-bltr-{}-{}.{}".format(str(image_name),str(d),str(sigma),str(image_ext))), spine_crop_blt)

    # spine_crop_blt = spine_crop_enh.copy()
    # spine_crop_blt_weak = cv2.bilateralFilter(spine_crop_enh, 13, 65, 65)
    spine_crop_blt_weak = cv2.bilateralFilter(spine_crop_enh, 13, 95, 95)
    dx=1
    dy=0
    ksize = 3
    ## when ddepth=-1 then the output image will have the same depth as the source
    sobel_x = cv2.Sobel(spine_crop_blt_weak, ddepth=-1, dx=dx, dy=dy, ksize=ksize, scale=2)
    cv2.imwrite(os.path.join(results_directory,"{}_05-sobel_x-{}-{}-{}.{}".format(str(image_name),str(dx),str(dy),str(ksize),str(image_ext))), sobel_x)

    spine_crop_edges = cv2.addWeighted(spine_crop_blt, 0.7, sobel_x, 0.3, 0)
    cv2.imwrite(os.path.join(results_directory,"{}_06-edge_ench.{}".format(str(image_name),str(image_ext))), spine_crop_edges)

    ### adaptive equalization for improving image contrast
    clip_limit = 3
    # tile_size_per = 0.76
    tile_size_per = 0.86 if algorithms_strength=="strong" else 0.5
    tile_size = (image_width//int(image_width*tile_size_per),image_height//int(image_height*tile_size_per))
    spine_crop_edges = pr.adaptive_equalization(spine_crop_edges, clip_limit=clip_limit, tile_size=tile_size)
    cv2.imwrite(os.path.join(results_directory,"{}_07-enh-contrast_{}_{}.{}".format(str(image_name),str(clip_limit),str(tile_size),str(image_ext))), spine_crop_edges)

    ### Find the Spine central line points
    central_line_points = compute.find_central_line(spine_crop_edges, algorithms_strength)
    print(f"Number of central_line_points: {len(central_line_points)}")
    # Display found line over the cropped spine
    point_radius = 3
    image_initial_clp = spine_crop.copy()
    image_initial_clp=cv2.circle(image_initial_clp, central_line_points[0].as_tuple(), point_radius, (0,0,255), 1)
    for i in range(1, len(central_line_points)):
        point = central_line_points[i]
        prev_point = central_line_points[i-1]
        # image_initial_clp=cv2.line(image_initial_clp, prev_point.as_tuple(), point.as_tuple(), (0,0,255), 1)
        image_initial_clp=cv2.circle(image_initial_clp, point.as_tuple(), point_radius, (0,0,255), 1)
    cv2.imwrite(os.path.join(results_directory,"{}_09-initial_clp.{}".format(str(image_name),str(image_ext))), image_initial_clp)

    ### Refine the Spine central line points
    epsilon = int(spine_width*0.1) if algorithms_strength == "strong" else int(spine_width*0.13)
    central_line_points_processed_1 = compute.refine_central_line_avg(central_line_points, epsilon)

    print(f"Refine with averaging filter {len(central_line_points_processed_1)} central_line_points")
    image_clp_1 = spine_crop.copy()
    image_clp_1 = cv2.circle(image_clp_1, central_line_points_processed_1[0].as_tuple(), point_radius, (0,0,255), 1)
    for i in range(1, len(central_line_points_processed_1)):
        point = central_line_points_processed_1[i]
        image_clp_1=cv2.circle(image_clp_1, point.as_tuple(), point_radius, (0,0,255), 1) #BGR
        
    cv2.imwrite(os.path.join(results_directory,"{}_10-clp_1_avg.{}".format(str(image_name),str(image_ext))), image_clp_1)


    central_line_points_processed_2 = compute.refine_central_line_hog(central_line_points_processed_1, spine_enh_image, algorithms_strength)
    print(f"Refine {len(central_line_points_processed_2)} central line points with HOG features")
    # Display found line over the cropped spine
    image_clp_2 = spine_crop.copy()
    image_clp_2 = cv2.circle(image_clp_2, central_line_points_processed_2[0].as_tuple(), point_radius, (0,0,255), 1)
    for i in range(1, len(central_line_points_processed_2)):
        point = central_line_points_processed_2[i]
        image_clp_2 = cv2.circle(image_clp_2, point.as_tuple(), point_radius, (0,0,255), 1) #BGR

    cv2.imwrite(os.path.join(results_directory,"{}_10-clp_2_hog.{}".format(str(image_name),str(image_ext))), image_clp_2)


    x_coord = [point.x for point in central_line_points_processed_2]
    x_coord_processed = median_filter(x_coord, size=3)
    central_line_points_processed_4 = [Point(x, p.y) for x, p in zip(x_coord_processed, central_line_points_processed_2)]
    print(f"Refine with median filter {len(central_line_points_processed_4)} central_line_points with epsilon {epsilon} ")

    image_clp_4 = spine_crop.copy() 
    image_clp_4 = cv2.circle(image_clp_4, central_line_points_processed_4[0].as_tuple(), point_radius, (0,0,255), 1)
    for i in range(1, len(central_line_points_processed_4)):
        point = central_line_points_processed_4[i]
        image_clp_4=cv2.circle(image_clp_4, point.as_tuple(), point_radius, (0,0,255), 1) #BGR
        
    cv2.imwrite(os.path.join(results_directory,"{}_10-clp_4_median.{}".format(str(image_name),str(image_ext))), image_clp_4)

    ### Convert data from central line points to pandas dataframe and apply ewm (exponentially weighted moving) smoothing
    df = pd.DataFrame(central_line_points_processed_4, columns =['x', 'y'])
    alpha = 0.1        # smoothing factor; higher value means less weight to recent observations
    df_smoothed = df.ewm(alpha=alpha).mean()
    
    image_clp_smoothed = spine_crop.copy()
    for idx in df_smoothed.index:
        if idx > 1:
            point_1 = (int(df_smoothed['x'][idx-1]), int(df_smoothed['y'][idx-1])) 
            point_2 = (int(df_smoothed['x'][idx]), int(df_smoothed['y'][idx]))
            image_clp_smoothed=cv2.line(image_clp_smoothed, point_1, point_2, (0,0,255), 1)    # red

    cv2.imwrite(os.path.join(results_directory,"{}_12-smoothed_{}.{}".format(str(image_name),str(alpha),str(image_ext))), image_clp_smoothed)

    ### Smooth smoothed data
    span = 15        # alpha = 2/(span + 1), span >= 1
    df_smoothed_1 = df_smoothed.ewm(span=span).mean()
    image_clp_smoothed_2 = spine_crop.copy()        
    for idx in df_smoothed_1.index:
        if idx>1:
            point_1 = (int(df_smoothed_1['x'][idx-1]), int(df_smoothed_1['y'][idx-1])) 
            point_2 = (int(df_smoothed_1['x'][idx]), int(df_smoothed_1['y'][idx]))
            image_clp_smoothed_2=cv2.line(image_clp_smoothed_2, point_1, point_2, (0,0,255), 1)
    
    cv2.imwrite(os.path.join(results_directory,"{}_13-smoothed-2_{}.{}".format(str(image_name),str(span),str(image_ext))), image_clp_smoothed_2)

    n = len(df)
    # smoothing = n - math.sqrt(2 * n)
    smoothing = n + (math.sqrt(2 * n) )
    ### Find line extremums and angles
    # xx, yy, extremums_x, extremums_y, lines_1, lines_2, max_angles, max_angles_2 = compute.compute_cob_angles(
    xx, yy, extremums_x, extremums_y, max_angles = compute.compute_cob_angles(
        df_smoothed_1,
        xb=0, xe=spine_height,
        spline_degree = 3,
        smoothing = smoothing
        )

    for ma in max_angles:
        print(f"Max angle at apex: ({ma.apex.y}, {ma.apex.x}) - {ma.measure}\N{DEGREE SIGN}C")
    # results[image_name] = [ma.measure for ma in max_angles]
    result = [(ma.apex.y, ma.apex.x, ma.measure) for ma in max_angles]

    # Draw spine curve
    end_line_image = spine_crop.copy()
    for i in range(1, len(xx)):
        point_1 = (int(yy[i-1]), int(xx[i-1]))
        point_2 = (int(yy[i]), int(xx[i]))
        
        end_line_image = cv2.line(end_line_image, point_1, point_2, (255,255,0), 1)    # BGR - cyan

    # Draw spine apices
    for i in range(len(extremums_x)):
        x = int(extremums_y[i])
        y = int(extremums_x[i])
        end_line_image = cv2.circle(end_line_image, (x, y), 4, (0,255,255), 2) # yellow
    cv2.imwrite(os.path.join(results_directory,"{}_14-apices_{}.{}".format(str(image_name),str(smoothing),str(image_ext))), end_line_image)
    

    # Plot apices, tangent lines and angle degrees
    fig, ax = plt.subplots()
    color_list_1 = ['red', 'green', 'gold', 'cyan', 'dodgerblue', 'violet', 'tomato']
    color_list_1 += color_list_1
    
    # plt.scatter(xs, ys, **{"color": "cyan", "marker": "."}, label="original")
    plt.scatter(
        extremums_y, extremums_x, **{"color": "orange", "marker": "o"}, label="Extremums"
    )
    plt.plot(yy, xx, **{"color": "blue", "ls": "-"}, label="Spine curve")


    # Draw tangent lines and angle degrees over cropped spine image
    end_line_image5 = end_line_image.copy()
    # BGR: rgb - cmyk
    colors=((0,0,255), (0,255,255), (255,0,255), (0,0,125), (0,0,255), (255,0,255), (0,255,255), (0,0,125), (0,0,255), (0,255,255), (255,0,255), (0,0,125), (0,0,255), (255,0,255), (0,255,255), (0,0,125))
    i = 0
    n = len(max_angles) if len(max_angles) < 5 else 10
    for element in max_angles[:n]:
        # print("Element: ", element)
        degree = int(element.measure)
        point_11 = (int(element.line_a.a.y), int(element.line_a.a.x))
        point_12 = (int(element.line_a.b.y), int(element.line_a.b.x))
        
        point_21 = (int(element.line_b.a.y), int(element.line_b.a.x))
        point_22 = (int(element.line_b.b.y), int(element.line_b.b.x))

        apex = (int(element.apex.y), int(element.apex.x))

        end_line_image5 = cv2.line(end_line_image5, point_21, point_22, colors[i], thickness=2, lineType=cv2.LINE_4)
        end_line_image5 = cv2.line(end_line_image5, point_11, point_12, colors[i], thickness=1, lineType=cv2.LINE_4)

        cv2.putText(end_line_image5, text=str(degree), org=(apex[0]-25, apex[1]), color=(0,255,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5) 
        
        plt.text(element.apex.y+10, element.apex.x+20, str(element.measure))
        plt.plot([point_11[0], point_12[0]], [point_11[1], point_12[1]], **{"color": color_list_1[i], "ls": "-"}, label=f"tangent_{i}")
        plt.plot([point_21[0], point_22[0]], [point_21[1], point_22[1]], **{"color": color_list_1[i], "ls": "-"}, label=f"tangent_{i}")
        i += 1

    cv2.imwrite(os.path.join(results_directory,"{}_15-cob.{}".format(str(image_name),str(image_ext))), end_line_image5)

    # Arrange spine curve plot
    plt.title("Spine curve and Cob angles")
    plt.ylabel('Image height')
    plt.xlabel('Image width')
    
    plt.ylim([-15, spine_height])  # range from 0 to crop_width
    plt.xlim([0, spine_width])
    
    ax = plt.gca()
    ax.invert_yaxis()

    plt.legend(loc="best", fancybox=True, shadow=True)
    fig.set_figheight(int(spine_height*1.3)//100)
    fig.set_figwidth(int(spine_width*4)//100)
    plt.savefig(os.path.join(results_directory,"{}_14-tangents.{}".format(str(image_name),"png")))
    plt.clf()
    plt.close()
    # plt.show()

    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    return result

if __name__ == '__main__':
    pass

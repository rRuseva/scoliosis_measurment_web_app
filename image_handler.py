import cv2
import os
import re 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pydicom as dicom
from io import BytesIO
import preprocessing as pr 
import computation as compute
from computation import Point, Line

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
    dataset = dicom.dcmread(BytesIO(file_content))
    print(dataset.is_implicit_VR)
    
    # remove sensitive information in DICOM metadata as patient Name, ID, age, birth date and sex
    dataset = anonymise_dicom_data(dataset=dataset)
    
    # dicom.filewriter.write_file(file_name, file_content, False)
    dicom.dcmwrite(filename=file_name, dataset=dataset, write_like_original=True)


def anonymise_dicom_data(dataset: dicom.FileDataset) ->  dicom.FileDataset:
    """Deletes sensitive data from DICOM type files to protect patient personal information.

    Args:
        dataset (dicom.FileDataset): Given DICOM image

    Returns:
        dicom.FileDataset: Same data set as the given one, but without Pattion information as ID, Name, Sex, Age and birthdate
    """
    print("Anonymise DICOM data ...")
    dataset.PatientID = None
    dataset.PatientName = None
    dataset.PatientSex = None
    dataset.PatientAge = None
    dataset.PatientBirthDate = None
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
        image_type =  ds.PhotometricInterpretation		# usually dicom x-ray is MONOCHROME2
        if ds.pixel_array.any():
            ### convert byte raw image data into uint8 in range [0,255]
            print("Converting byte raw data from dicom into uint8")
            image_data = ds.pixel_array - np.min(ds.pixel_array)
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

    if original_image_height > 1500:
        scale_ratio = 0.2
    else:
        scale_ratio = 1

    image_height = int(original_image_height*scale_ratio)
    image_width = int(original_image_width*scale_ratio)

    grey_image = cv2.resize(grey_image, (image_width,image_height), interpolation = cv2.INTER_AREA)
    image = cv2.resize(original_image, (image_width,image_height)  , interpolation = cv2.INTER_AREA)
    print("Scale ratio: {} \nNew image size: {}-{}".format(scale_ratio, image_width, image_height))

    cv2.imwrite(os.path.join(results_directory,"{}_00-original_image.{}".format(str(image_name),str(image_ext))), image)
    
    
    ### auto image enhancement for improving brightness and contrast - histogram stretching
    ### not enough
    clip_hist_percent = 15
    enh_image, alpha, beta = pr.automatic_brightness_and_contrast(grey_image, clip_hist_percent=clip_hist_percent)
    cv2.imwrite(os.path.join(results_directory,"{}_01-enh-1_{}.{}".format(str(image_name),str(clip_hist_percent),str(image_ext))), enh_image)
    
    ### adaptive equalization for improving image contrast 
    # clip_limit = 3
    # tile_size_per = 0.46
    # tile_size = (image_width//int(image_width*tile_size_per),image_height//int(image_height*tile_size_per))
    # enh_image = pr.adaptive_equalization(grey_image, clip_limit=clip_limit, tile_size=tile_size)
    # cv2.imwrite(os.path.join(results_directory,"{}_03-enh-1_{}.{}".format(str(image_name),str(tile_size),str(image_ext))), enh_image)
    
    ### Detecting spine ROI
    sum_col, sum_row = pr.intensity_projection(enh_image)
    spine_start, spine_end, col_values, min_max_row = pr.detect_spine(enh_image, sum_col, sum_row)
    print("Cropped pos: {}-{}".format(spine_start, spine_end))

    ### Plot intensity projection histograms and detected spine ROI
    fig = plt.figure()
    plt.suptitle("Intensity projection")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Vertical')
    ax2.title.set_text('Horizontal')

    plt.subplot(1, 2, 1)
    plt.bar(range(0,image_width), sum_col, align='edge', width=1.0, color='coral')
    plt.axvline(x=spine_start[0], color='cyan')
    plt.axvline(x=spine_end[0], color='red')
    plt.ylabel('Intensity' )
    plt.xlabel('Image width')


    plt.subplot(1, 2, 2)
    y_ax = np.arange(image_height)
    plt.barh(np.arange(image_height), sum_row, align='center', height=1.0, color='turquoise')
    plt.barh(np.arange(image_height), min_max_row, align='center', height=1.0, color='paleturquoise')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.axhline(y=spine_start[1], color='coral')
    plt.axhline(y=spine_end[1], color='red')
    plt.ylabel('Image height')
    plt.xlabel('Intensity')
    fig.tight_layout(pad=1.0)

    plt.savefig(os.path.join(results_directory,"{}_02-intensity_projection.{}".format(str(image_name),"png")))
    plt.clf()


    ### Cropping spine region
    spine_crop_enh = enh_image[spine_start[1]:spine_end[1], spine_start[0]:spine_end[0]]
    spine_crop = image[spine_start[1]:spine_end[1], spine_start[0]:spine_end[0]]
    spine_crop_grey = grey_image[spine_start[1]:spine_end[1], spine_start[0]:spine_end[0]]
    
    spine_height, spine_width = spine_crop_grey.shape
    print("Cropped spine size: {}".format(spine_crop_grey.shape))
    cv2.imwrite(os.path.join(results_directory,"{}_02-spine-crop.{}".format(str(image_name),str(image_ext))), spine_crop_grey)


    ### Denoising - apply strong bilateral filter
    d = 15
    sigma = 115
    spine_crop_blt = cv2.bilateralFilter(spine_crop_enh, d, sigma, sigma)
    cv2.imwrite(os.path.join(results_directory,"{}_03-bltr-{}-{}.{}".format(str(image_name),str(d),str(sigma),str(image_ext))), spine_crop_blt)


    ### Find the Spine central line points
    central_line_points = compute.find_central_line(spine_crop_blt)
    print(f"Number of central_line_points: {len(central_line_points)}")
    # Display found line over the cropped spine
    image_initial_clp = spine_crop.copy()
    for i in range(1, len(central_line_points)):
        point = central_line_points[i]
        prev_point = central_line_points[i-1]
        image_initial_clp=cv2.line(image_initial_clp, prev_point.as_tuple(), point.as_tuple(), (0,0,255), 1)
    cv2.imwrite(os.path.join(results_directory,"{}_09-initial_clp.{}".format(str(image_name),str(image_ext))), image_initial_clp)

    ### Refine the set of central line points
    # epsilon = 10
    epsilon = 13*spine_width//100
    central_line_points_processed = compute.refine_central_line(central_line_points, epsilon)
    image_clp = spine_crop.copy()
    # Display refined line over the cropped spine
    for i in range(1, len(central_line_points_processed)):
        point = central_line_points_processed[i]
        prev_point = central_line_points_processed[i-1]
        image_clp=cv2.line(image_clp, prev_point.as_tuple(), point.as_tuple(), (0,0,255), 1)
    cv2.imwrite(os.path.join(results_directory,"{}_10-clp-{}.{}".format(str(image_name),str(epsilon),str(image_ext))), image_clp)


    ### Convert data from central line points to pandas dataframe and apply ewm (exponentially weighted moving) smoothing
    df = pd.DataFrame(central_line_points_processed, columns =['x', 'y'])

    alpha = 0.1		# smoothing factor; higher value means less weight to recent observations
    df_smoothed = df.ewm(alpha=alpha).mean()
    
    image_clp_smoothed = spine_crop.copy()
    for idx in df_smoothed.index:
        if idx > 1:
            point_1 = (int(df_smoothed['x'][idx-1]), int(df_smoothed['y'][idx-1])) 
            point_2 = (int(df_smoothed['x'][idx]), int(df_smoothed['y'][idx]))
            
            image_clp_smoothed=cv2.line(image_clp_smoothed, point_1, point_2, (0,0,255), 1)	# red
    cv2.imwrite(os.path.join(results_directory,"{}_12-smoothed_{}.{}".format(str(image_name),str(alpha),str(image_ext))), image_clp_smoothed)

    ### Smooth smoothed data
    span = 15 		# alpha = 2/(span + 1), span >= 1
    df_smoothed_1 = df_smoothed.ewm(span=span).mean()
    image_clp_smoothed_2 = spine_crop.copy()		
    for idx in df_smoothed_1.index:
        if idx>1:
            point_1 = (int(df_smoothed_1['x'][idx-1]), int(df_smoothed_1['y'][idx-1])) 
            point_2 = (int(df_smoothed_1['x'][idx]), int(df_smoothed_1['y'][idx]))

            image_clp_smoothed_2=cv2.line(image_clp_smoothed_2, point_1, point_2, (0,0,255), 1)
    
    cv2.imwrite(os.path.join(results_directory,"{}_13-smoothed-2_{}.{}".format(str(image_name),str(alpha),str(image_ext))), image_clp_smoothed_2)


    ### Find line extremums and angles
    # xx, yy, extremums_x, extremums_y, lines_1, lines_2, max_angles, max_angles_2 = compute.compute_cob_angles(
    xx, yy, extremums_x, extremums_y, max_angles = compute.compute_cob_angles(
        df_smoothed_1,
        xb=0, xe=spine_height,
        spline_degree = 5
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
        
        end_line_image = cv2.line(end_line_image, point_1, point_2, (255,255,0), 1)	# BGR - cyan

    # Draw spine apices
    for i in range(len(extremums_x)):
        x = int(extremums_y[i])
        y = int(extremums_x[i])
        end_line_image = cv2.circle(end_line_image, (x, y), 4, (0,255,255), 2) # yellow
        
    cv2.imwrite(os.path.join(results_directory,"{}_14-apices_{}.{}".format(str(image_name),str(alpha),str(image_ext))), end_line_image)
    

    # Plot apices, tangent lines and angle degrees
    fig, ax = plt.subplots()
    color_list_1 = ['red', 'green', 'gold', 'cyan', 'dodgerblue', 'violet', 'tomato']
    color_list_1 += color_list_1
    
    # plt.scatter(ys, xs, **{"color": "cyan", "marker": "."}, label="original")
    plt.scatter(
        extremums_y, extremums_x, **{"color": "orange", "marker": "o"}, label="Extremums"
    )
    plt.plot(yy, xx, **{"color": "blue", "ls": "-"}, label="Spine curve")


    # Draw tangent lines and angle degrees over cropped spine image
    end_line_image5 = end_line_image.copy()
    # BGR: rgb - cmyk
    colors=((0,0,255), (0,255,255), (255,0,255), (0,0,125), (0,0,255), (255,0,255), (0,255,255), (0,0,125))
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

    plt.ylim([0, spine_height])  # range from 0 to crop_width
    plt.xlim([0, spine_width]) 
    # plt.xlim([yy[0]-spine_width//4, yy[-1]+spine_width//4])  # range from 0 to crop_height
    
    ax = plt.gca()
    ax.invert_yaxis()

    # plt.axis("equal")
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.savefig(os.path.join(results_directory,"{}_14-tangents.{}".format(str(image_name),"png")))
    plt.clf()
    plt.close()
    # plt.show()

    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    return result

if __name__ == '__main__':
    pass

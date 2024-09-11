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
import computation 

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
    print("Saving file content as DICOM file ...")
    print(type(file_content))
    dataset = dicom.dcmread(BytesIO(file_content))
    print(dataset.is_implicit_VR)
    
    # remove sensitive information in DICOM metadata as patient Name, ID, age, birth date and sex
    dataset = anonymise_dicom_data(dataset=dataset)
    
    # dicom.filewriter.write_file(file_name, file_content, False)
    dicom.dcmwrite(filename=file_name, dataset=dataset, write_like_original=True)


def anonymise_dicom_data(dataset: dicom.FileDataset) ->  None:
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
            print(f"Converting byte raw data from dicom into uint8")
            image_data = ds.pixel_array - np.min(ds.pixel_array)
            image_data = image_data / np.max(image_data)
            image_data = (image_data * 255).astype(np.uint8)        

    if image_data.any():
        print(f"image data: {image_data}")
        return (image_data, image_name, image_ext, image_type)
    else:
        print(f"image data is missing")


def process_image(filename, image_directory, results_directory):
    original_image, image_name, image_ext, image_type = open_image_file(filename, image_directory)
    
    print("Processing: {} - {} - {}".format(image_name, original_image.shape, image_type))
    if image_type != "MONOCHROME2":
        grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        grey_image = original_image

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
    print("scale ratio: {} \nnew image size: {}-{}".format(scale_ratio, image_width, image_height))
    cv2.imwrite(os.path.join(results_directory,"{}_00-original_image.{}".format(str(image_name),str(image_ext))), image)


    ### auto image enhancment for improving brightness and contrast - histogram strching
    ### not enough
    clip_hist_percent = 15
    enh_image, alpha, beta = pr.automatic_brightness_and_contrast(grey_image, clip_hist_percent=clip_hist_percent)
    cv2.imwrite(os.path.join(results_directory,"{}_01-enh-1_{}.{}".format(str(image_name),str(clip_hist_percent),str(image_ext))), enh_image)

    ### adaptive equalization for improving image contrast 
    # clip_limit = 3
    # tile_size_per = 0.46
    # tile_size = (image_width//int(image_width*tile_size_per),image_height//int(image_height*tile_size_per))
    # enh_image = pr.adaptive_equalization(grey_image, clip_limit=clip_limit, tile_size=tile_size)
    # cv2.imwrite(os.path.join(results_directory,"{}_00-enh-1_{}.{}".format(str(image_name),str(tile_size),str(image_ext))), enh_image)

    ### detecting spine ROI
    sum_col, sum_row = pr.intensity_projection(enh_image)
    spine_start, spine_end, col_values, min_max_row = pr.detect_spine(enh_image, sum_col, sum_row)
    print("Cropped pos: {}-{}".format(spine_start, spine_end))

    fig = plt.figure()
    plt.suptitle("Intensity projection")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Vertical')
    ax2.title.set_text('Horizontal')
    plt.subplot(1, 2, 1)
    # plt.suptitle("Vertical Intensity projection")
    plt.bar(range(0,image_width), sum_col, align='edge', width=1.0, color='coral')
    plt.axvline(x=spine_start[0], color='cyan')
    plt.axvline(x=spine_end[0], color='red')

    plt.ylabel('intensity' )
    plt.xlabel('image width')
    # plt.savefig(os.path.join(results_directory,"{}_00-intensity_projection-vertical.{}".format(str(image_name),"png")))
    # plt.clf()

    plt.subplot(1, 2, 2)
    # plt.suptitle("Horizontal Intensity projection")
    # fig = plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
    y_ax = np.arange(image_height)
    # plt.bar(range(0,image_height), sum_row, align='edge', width=1.0, color='cyan')
    plt.barh(np.arange(image_height), sum_row, align='center', height=1.0, color='turquoise')
    plt.barh(np.arange(image_height), min_max_row, align='center', height=1.0, color='paleturquoise')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.axhline(y=spine_start[1], color='coral')
    plt.axhline(y=spine_end[1], color='red')
    # plt.ylabel('image height')
    plt.xlabel('intensity')

    # plt.subplots(1, 2, figsize=(8, 8))
    # plt.savefig(os.path.join(results_directory,"{}_00-intensity_projection-horizontal.{}".format(str(image_name),"png")))
    plt.savefig(os.path.join(results_directory,"{}_00-intensity_projection.{}".format(str(image_name),"png")))
    plt.clf()



    ### cropping spine region
    spine_crop_enh = enh_image[spine_start[1]:spine_end[1], spine_start[0]:spine_end[0]]
    spine_crop = image[spine_start[1]:spine_end[1], spine_start[0]:spine_end[0]]
    spine_crop_grey = grey_image[spine_start[1]:spine_end[1], spine_start[0]:spine_end[0]]

    spine_height, spine_width = spine_crop_grey.shape
    print("Cropped spine size: {}".format(spine_crop_grey.shape))
    cv2.imwrite(os.path.join(results_directory,"{}_01-spine-crop.{}".format(str(image_name),str(image_ext))), spine_crop_grey)

    ### denoising - bilateral filter
    d = 15
    sigma = 115
    # spine_crop_enh = cv2.bilateralFilter(spine_crop_grey, d, sigma, sigma)
    spine_crop_blt = cv2.bilateralFilter(spine_crop_enh, d, sigma, sigma)
    # spine_crop_blt = cv2.bilateralFilter(spine_crop_grey, d+2, sigma+2, sigma+2)
    cv2.imwrite(os.path.join(results_directory,"{}_03-bltr-{}-{}.{}".format(str(image_name),str(d),str(sigma),str(image_ext))), spine_crop_blt)

    # image_h, image_w = combined.shape
    image_h, image_w = spine_crop.shape
    window_w = 60
    # window_w = int(image_w*0.33)
    window_h = 10
    step_w = 1
    step_h = 5
    c_y = c_x = 0
    # (sum, x, y)
    # max_sum = (0, 0, 0)
    central_line_points = []

    while c_y < image_h - window_h:
        max_sum = max_sum = (0, 0, 0)
        c_x = 0
        while c_x < image_w - window_w:
            roi = spine_crop_blt[c_y:c_y+window_h, c_x:c_x+window_w]
            current_sum  = np.sum(roi)
            if max_sum[0] < current_sum:
                curr_x = c_x+window_w//2
                
                max_sum = (current_sum, curr_x, c_y)
            c_x += step_w
        central_line_points.append((max_sum[1],max_sum[2]))
        c_y += step_h


    print(f"Number of central_line_points: {len(central_line_points)}")

    # TODO - add restraining in width 

    end_line_image = spine_crop.copy()
    # epsilon = int(window_w*0.25)
    epsilon = int(image_width*0.02)
    prev_point = image_width//2

    i = 0

    for point in central_line_points:
        curr_x = point[0]
        curr_y = point[1]

        diff = curr_x - prev_point
        if abs(diff) > epsilon and i > 1:
            # curr_x = curr_x + diff//2
            curr_x = prev_point
        point = (curr_x, curr_y)
        prev_point = curr_x
        i += 1	
                
        end_line_image=cv2.line(end_line_image,(point[0], point[1]), (point[0], point[1]+step_h), (0,0,255), 1)

    cv2.imwrite(os.path.join(results_directory,"{}_11-cl.{}".format(str(image_name),str(image_ext))), end_line_image)

    # from statsmodels.tsa.api import SimpleExpSmoothing

    df = pd.DataFrame(central_line_points, columns =['x', 'y'])
    # print(df)
    alpha = 0.1
    df_smoothed = df.ewm(alpha=alpha).mean()
    alpha = 15
    # df_smoothed = df.ewm(span=alpha).mean()
    # print(df_smoothed)

    end_line_image2 = spine_crop.copy()
    end_line_image2 = cv2.cvtColor(end_line_image2, cv2.COLOR_GRAY2BGR)
    for idx in df_smoothed.index:
        # print(float(df_smoothed['x'][idx]))
        if idx > 1:
            x1 = int(df_smoothed['x'][idx-1])
            y1 = int(df_smoothed['y'][idx-1])
            x2 = int(df_smoothed['x'][idx])
            y2 = int(df_smoothed['y'][idx])
            end_line_image2=cv2.line(end_line_image2,(x1, y1), (x2, y2), (0,0,255), 1)	# red


    end_line_image20 = spine_crop.copy() 	
    end_line_image20 = cv2.cvtColor(end_line_image20, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(os.path.join(results_directory,"{}_12-smoothed_{}.{}".format(str(image_name),str(alpha),str(image_ext))), end_line_image2)

    df_smoothed_1 = df_smoothed.ewm(span=alpha).mean()
    # print(df_smoothed)

    end_line_image3 = spine_crop.copy()		
    for idx in df_smoothed_1.index:
        # print(df_smoothed['x'][idx])
        if idx>1:
            x1 = int(df_smoothed_1['x'][idx-1])
            y1 = int(df_smoothed_1['y'][idx-1])
            x2 = int(df_smoothed_1['x'][idx])
            y2 = int(df_smoothed_1['y'][idx])
            # print(x2)
            end_line_image3=cv2.line(end_line_image3,(x1, y1), (x2, y2), (0,0,255), 1)
    
    cv2.imwrite(os.path.join(results_directory,"{}_13-smoothed-2_{}.{}".format(str(image_name),str(alpha),str(image_ext))), end_line_image3)
    



    xx, yy, extremums_x, extremums_y, lines_1, lines_2, max_angles, max_angles_2 = computation.compute_cob_angles(
        df_smoothed_1,
        xb=0, xe=image_h,
        spline_degree = 5
        )

    end_line_image4 = spine_crop.copy()
    end_line_image4 = cv2.cvtColor(end_line_image4, cv2.COLOR_GRAY2BGR)
    for i in range(1, len(xx)):
        # if i > 1:
        x1 = int(yy[i-1])
        y1 = int(xx[i-1])
        x2 = int(yy[i])
        y2 = int(xx[i])
        end_line_image4=cv2.line(end_line_image4,(x1, y1), (x2, y2), (255,255,0), 2)

    for i in range(len(extremums_x)):
        x = int(extremums_y[i])
        y = int(extremums_x[i])
        end_line_image4=cv2.circle(end_line_image4,(x, y), 4, (0,255,255), 2)
        
    cv2.imwrite(os.path.join(results_directory,"{}_14-cob_{}.{}".format(str(image_name),str(alpha),str(image_ext))), end_line_image4)
    end_line_image5 = end_line_image4.copy()
    # end_line_image5 = cv2.cvtColor(end_line_image5, cv2.COLOR_GRAY2BGR)
    colors=((0,0,255), (255,0,255), (0,255,255), (0,0,125), (0,0,255), (255,0,255), (0,255,255), (0,0,125))
    i=0
    n = len(max_angles_2) if len(max_angles_2)<5 else 10
    for element in max_angles_2[:n]:
        
        degree = int(element[0])
        y11 = int(element[1][0])
        x11 = int(element[1][1])
        y12 = int(element[1][2])
        x12 = int(element[1][3])

        y21 = int(element[2][0])
        x21 = int(element[2][1])
        y22 = int(element[2][2])
        x22 = int(element[2][3])

        apex = (int(element[3][0]), int(element[3][1]))
        print(f"tangent line 1: {x11}, {y11} - {x12}, {y12}")
        print(f"tangent line 2: {x21}, {y21} - {x22}, {y22}")
        end_line_image5=cv2.line(end_line_image5,(x21, y21), (x22, y22), colors[i], 2)
        end_line_image5=cv2.line(end_line_image5,(x11, y11), (x12, y12), colors[i], 2)
        cv2.putText(end_line_image5, text=str(degree), org=(apex[1]-25, apex[0]), color=(0,255,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5) 
        i += 1
    cv2.imwrite(os.path.join(results_directory,"{}_15-cob_{}.{}".format(str(image_name),str(alpha),str(image_ext))), end_line_image5)
    

    fig, ax = plt.subplots()
    # plt.scatter(xs, ys, **{"color": "cyan", "marker": "."}, label="original")
    plt.scatter(
        extremums_y, extremums_x, **{"color": "orange", "marker": "o"}, label="Extremums"
    )
    # print(f"{np.flip(xx)[:5]}")
    # plt.plot(np.flip(yy), xx, **{"color": "blue", "ls": "-"}, label="B-spline")
    plt.plot(yy, xx, **{"color": "blue", "ls": "-"}, label="B-spline")
    # plt.plot(xx,yy1, **{'color': 'red', 'ls': '-.'}, label="B-spline deriv")

    extremums_x = extremums_x[1:-1]
    extremums_y = extremums_y[1:-1]
    for i in range(len(extremums_y)):
        print(extremums_x[i], extremums_y[i], max_angles[i])
        plt.text(extremums_x[i], extremums_y[i], max_angles[i][1])

    print("lines")
    color_list_1 = ['red', 'blue', 'gold', 'cyan', 'dodgerblue', 'violet', 'tomato']
    color_list_1 += color_list_1

    for i in range(len(lines_1)):
        print(f" {i}: {lines_1[i]} - {lines_2[i]}")
        plt.plot(lines_1[i][1], lines_1[i][0], **{"color": color_list_1[i], "ls": "-"}, label=f"tangent_{i}")
        plt.plot(lines_2[i][1], lines_2[i][0], **{"color": color_list_1[i], "ls": "-"}, label=f"tangent_{i}")

    plt.ylim([0, image_h])  # range from 0 to crop_width
    plt.xlim([0, image_w])  # range from 0 to crop_height
    ax = plt.gca()
    ax.invert_yaxis()

    plt.axis("equal")
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.savefig(os.path.join(results_directory,"{}_14-tangents.{}".format(str(image_name),"png")))
    plt.clf()


if __name__ == '__main__':
    pass
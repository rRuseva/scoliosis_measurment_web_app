# 🩻 Scoliosis Measurement Web Application

A Python application for analyzing spinal radiographs via image processing thenniques and automatically estimating the Cobb angle used in scoliosis assessment.

The project was developed as part of my Master's thesis and explores how classical digital image processing and numerical methods can be combined to extract the shape of the spine from an X-ray image and derive quantitative measurements of spinal curvature.

The image-processing pipeline is integrated into a Flask web application, allowing radiographs to be uploaded, processed, and the resulting measurements visualized through a web interface.

## Project Motivation

Scoliosis is characterized by an abnormal lateral curvature of the spine. One of the primary quantitative measurements used when evaluating scoliosis is the Cobb angle, which is traditionally determined manually from spinal radiographs. Manual measurements depend on the selection of anatomical reference points and can therefore be affected by observer variability.

The goal of this project was to investigate classical computer vision and image processing techniques for building an automated measurement pipeline capable of:

+ locating the spinal region in a radiograph
+ extracting a representation of the spinal centerline
+ analyzing the geometry of the detected spinal curve
+ identifying individual scoliotic curvatures
+ estimating their Cobb angles
+ visualizing the detected curve and measurements

**Note:** This project is an academic prototype developed for research and educational purposes and is not intended for clinical use.

## Tech stack
- Backend: Python, Flask, Flask-WTF, Waitress (WSGI server)
- Image processing: OpenCV, scikit-image, SciPy, NumPy
- Medical imaging: pydicom, python-gdcm (DICOM reading/writing & anonymization)
- Data & visualization: pandas, matplotlib

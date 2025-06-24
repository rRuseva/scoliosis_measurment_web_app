import math
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PPoly, splder, splev, splrep, make_splrep, sproot
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
import cv2


@dataclass
class Point:
	x: float
	y: float

	def as_tuple(self):
		return(self.x, self.y)

	def inverse(self):
		return Point(x=self.y, y=self.x)


@dataclass
class Line:
	a: Point
	b: Point

	def as_tuple(self):
		return((self.a.x, self.a.y), (self.b.x, self.b.y))


@dataclass
class TangentLine:
	slope: float
	coefficient: float

	def as_tuple(self):
		return(self.slope, self.coefficient)


@dataclass
class CobAngle:
	apex: Point
	line_a: Line
	line_b: Line
	measure: float


@dataclass
class AngleBetweenTangents:
	point_a: Point
	t_line_a: TangentLine
	point_b: Point
	t_line_b: TangentLine
	measure: float


def compute_cob_angles(points: list, xb, xe, smoothing, spline_degree = 5):
	"""_summary_

	Args:
		points (list[Point]): list of central line points to be fitted into a curve
		xb (float): lower limit
		xe (float): upper limit 
		spline_degree (int, optional): degree of the B-spline. Defaults to 5.
	"""
	print("Computing Cob angle ...")

	# Swap x and y coordinates
	ys = [int(points['x'][i]) for i in points.index]
	xs = [int(points['y'][i]) for i in points.index]
	n = len(ys)

	# ### Define B-spline representation of central line points representing the spine curve
	spine_curve = make_splrep(xs, ys, k=spline_degree, s=smoothing)
	# Define first derivative equation of the spline representation of the spine curve 
	spine_der = splder(spine_curve, n=1)

	# Construct evenly spaced samples, calculated over the interval for displaying b-spline curve
	xx, xx_step = np.linspace(xs[0], xs[-1], xs[-1]-xs[0], retstep=True)

	yy = splev(xx, spine_curve)  # splev Evaluate the B-spline over xx for displaying
	yy1 = splev(xx, spine_der)  # splev Evaluate the B-spline first order derivative over xx for displaying

	# ### Find the local extremums (minimums and maximums) of the spine curve, by constructing piecewise polinomial from the
	# B-spline object of the first derivative and evaluating its roots
	ppoly = PPoly.from_spline(spine_der)
	# discontinuity - whether to report sign changes across discontinuities at breakpoints as roots
	# extrapolate - whether to return roots from polynomial extrapolated based on first and last intervals
	extremums_x = ppoly.roots(discontinuity=False, extrapolate=False)
	extremums_x = np.sort(extremums_x)

	extremums_x = np.insert(arr=extremums_x, obj=0, values=xx[0])
	extremums_x = np.insert(arr=extremums_x, obj=len(extremums_x), values=xx[-1])

	extremums_y = splev(extremums_x, spine_curve)
	# print(f"The number of extremums: ({extremums_x.shape[0]}, {extremums_y.shape[0]})")

	len_extremums = len(extremums_x)
	max_angles = []
	min_x = min(xs)
	max_x = max(xs)
	epsilon = (max_x-min_x)//2

	# for each extremum	look at both sides and collect tangent lines as tuple of [(x, y), (slope, coefficient)]
    # for each combination of tangents calculate the angle and find the maximum
	for k in range(1, len_extremums-1):
		# slopes: tuple(Point, TangentLine)
		slopes_1 = [(Point(point_x, splev(point_x, spine_curve)),
					  tangent_line(curve_derivative=spine_der, point=Point(point_x, splev(point_x, spine_curve)) ) ) for point_x in np.linspace(start=extremums_x[k-1], stop=extremums_x[k], num=len(range(int(extremums_x[k-1]), int(extremums_x[k]))) )]
		
		slopes_2 = [(Point(point_x, splev(point_x, spine_curve)),
					  tangent_line(curve_derivative=spine_der, point=Point(point_x, splev(point_x, spine_curve)) ) ) for point_x in np.linspace(start=extremums_x[k], stop=extremums_x[k+1], num=len(range(int(extremums_x[k]), int(extremums_x[k+1]))) )]
		angles = []
		for i in range(len(slopes_1)):
			for j in range(len(slopes_2)):
				point_1, t_line_1 = slopes_1[i]
				point_2, t_line_2 = slopes_2[j]
				# angle_rad = compute_angle_from_slopes(slope_1, slope_2, True)
				# angle: ( (x1,y1, slope1, c1), (x2,y2, slope2, c2), radian)
				# angles.append(((x_1, y_1, slope_1, c_1), (x_2, y_2, slope_2, c_2), angle_rad))  
				angle_rad = compute_angle_from_slopes(t_line_1.slope, t_line_2.slope, in_rad=True)
				angles.append(AngleBetweenTangents(point_1, t_line_1, point_2, t_line_2, angle_rad) )

		# print(f"For extremum[{k}] found {len(angles)} angles")
		if len(angles) > 0:
			max_angle = max(angles, key=lambda x: x.measure)
			# print(f"For extremum[{k}] max angle is: {max_angle}")

			# epsilon = epsilon//2
			min_x = extremums_x[k] - epsilon
			max_x = extremums_x[k] + epsilon

			# compute the y-coordinate of the line ends
			y_min_1 = point_eq(min_x, max_angle.t_line_a.slope, max_angle.t_line_a.coefficient)
			y_max_1 = point_eq(max_x, max_angle.t_line_a.slope, max_angle.t_line_a.coefficient)

			y_min_2 = point_eq(min_x, max_angle.t_line_b.slope, max_angle.t_line_b.coefficient)
			y_max_2 = point_eq(max_x, max_angle.t_line_b.slope, max_angle.t_line_b.coefficient)
			

			line_1 = Line(max_angle.point_a, Point(max_x, y_max_1))
			line_2 = Line(max_angle.point_b, Point(min_x, y_min_2))


			max_angles.append(CobAngle(apex=Point(extremums_x[k], extremums_y[k]), 
							   		     line_a=line_1, line_b=line_2, measure=rad_to_deg(max_angle.measure)))

		else:
			np.delete(extremums_x, k)
			np.delete(extremums_y, k)

	return(xx, yy, extremums_x, extremums_y, max_angles)


def tangent_line(curve_derivative: tuple, point: Point) -> tuple[np.ndarray, np.float64]:
    """Computes the slope and the coefficient of the tangent line at given point.
    The tangent line at certain point can be described with its slope and coefficient, this coming from the line equation: 
    y = slope * x + coefficient 
    Where the slope can be derived from the first order derivative of the curve at this point: slope = curve derivative at point x
    Therefor the coefficient = y - slope * x

    Args:
        curve_der (spline): The derivative of the curve
        point (tuple(numpy.float64, numpy.float64)): given x and y of a point

    Returns:
        tuple[np.ndarray, np.float64]: tuple of the slope and the coefficient
	"""
    slope = splev(x=point.x, tck=curve_derivative)
    b = point.y - slope * point.x

    return TangentLine(slope, b)

 
def point_eq(x: int, slope: np.ndarray, coef: np.float64) -> np.float64:
    """Find the y coordinate of point x from line given as slope and coefficient by the 'point-slope' formula.
    y = slope * x + coef

    Args:
        x (int): x coordinate
        slope (np.ndarray): the slope ??? why it is np.ndarray
        coef (np.float64): the coefficient

    Returns:
        np.float64: y value from a line at x
	"""
    return slope * x + coef


def rad_to_deg(rad: float) -> float:
    """Convert radians into degrees

    Args:
        rad (float): angle in radians

    Returns:
        float: angle in degrees
    """
    return 180.0 / math.pi * rad


def compute_angle_from_slopes(ma: np.ndarray, mb: np.ndarray, in_rad: bool) -> float:
    """Computes the angle between two tangent lines given by their slopes;

    Args:
        ma (np.ndarray): The slope of line a
        mb (np.ndarray): The slope of line b
        in_rad (bool): if false returns the result in degrees; otherwise in radians;

    Returns:
        float: angle
    """
    angle_rad = math.atan(abs((ma-mb)/(1+ma*mb)))

    if not in_rad:
        return rad_to_deg(angle_rad)
    
    return angle_rad


def find_central_line(spine_crop: np.ndarray, algorithms_strength:str) : #-> list[Point]:
	# Slide a window with size (window_w, window_h) over the image with horizontal step step_w
	# and vertical step step_h. At each step calculate the sum of pixel intensities. 
	# At each row identify the window with tha maximum sum and save the central point.
	# Return: List with all central points

	print("Finding central line points...")
	image_h, image_w = spine_crop.shape
	window_w = int(image_w*0.4) if algorithms_strength == "strong" else int(image_w*0.33)
	window_h = 16
	step_w = 1
	step_h = 6
	small_window_w = int(image_w*0.33)
	# current position of the window
	c_y = c_x = 0

	central_line_points = []
	i = 0
	while c_y < image_h - window_h:
		# (the maximum sum, the central point of the window with the maximum intensity)
		max_sum = (0, Point(0, 0))
		if len(central_line_points )> 0:
			prev_point_x = central_line_points[-1].x
			c_x = prev_point_x - small_window_w
		else:
			prev_point_x = 0
			c_x = 0

		# prev_point_x = 0
		# c_x = 0
		# while c_x < image_w - window_w:
		while c_x < prev_point_x + small_window_w and c_x < image_w - window_w:
			roi = spine_crop[c_y:c_y+window_h, c_x:c_x+window_w]
			current_sum  = np.sum(roi)
			if max_sum[0] < current_sum:
				curr_x = c_x + window_w//2
				# curr_y = c_y + window_h//2
				curr_y = c_y
				max_sum = (current_sum, Point(curr_x, curr_y))
			c_x += step_w
		central_line_points.append(max_sum[1])
		c_y += step_h
	return central_line_points


# refine based on hog features of two mirrored rectangles on both sides of central line point
def refine_central_line_hog(central_line_points, spine_crop: np.ndarray, algorithms_strength: str) -> list[Point]:
	print(f"Refine {len(central_line_points)} central line points via HOG features...")
	image_h, image_w = spine_crop.shape
	window_w = int(image_w*0.4) if algorithms_strength == "strong" else int(image_w*0.33)
	window_w = window_w - 1 if (window_w % 2 != 0) else window_w
	small_window_w = window_w//2
	small_window_w = small_window_w - 1 if small_window_w % 2 != 0 else small_window_w
	window_h = 16

	central_line_points_processed = []
	prev_x = central_line_points[0].x
	step_w = 1
	i = 1
	c_y = c_x = 0
	for point in central_line_points:
		prev_point = central_line_points[i-1]
		c_y = point.y

		roi_left = spine_crop[c_y:c_y+window_h, point.x-small_window_w:point.x]
		roi_right = spine_crop[c_y:c_y+window_h, point.x:point.x+small_window_w]	

		if roi_left.shape[1] < 5 or roi_right.shape[1] < 5:
			initial_similarity = 0
		else:
			hog_left = hog(roi_left, orientations=4, pixels_per_cell=(4, 4),
	           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
			hog_right_flipped = hog(np.flip(roi_right, axis=1), orientations=4, pixels_per_cell=(4, 4),
	           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

			initial_similarity = cosine_similarity([hog_left], [hog_right_flipped])
			initial_similarity = initial_similarity[0][0]

		# print(f"initial similarity: {initial_similarity}")
		if initial_similarity > 0.6:
			central_line_points_processed.append(point)
		else:
			c_x = point.x - small_window_w
			max_similarity = initial_similarity
			best_match = Point(point.x, c_y)
			while c_x < image_w - window_w:
				l_x = c_x - small_window_w if c_x - small_window_w > 0 else point.x
				r_x = c_x + small_window_w if c_x + small_window_w < image_w else image_w-small_window_w
				
				roi_left = spine_crop[c_y:c_y+window_h, l_x:l_x+small_window_w]
				
				roi_right = spine_crop[c_y:c_y+window_h, r_x:r_x+small_window_w]
				
				if roi_left.shape[1] < 5 or roi_right.shape[1] < 5:
					similarity = 0
				elif roi_left.shape[1] != roi_right.shape[1]:
					similarity = 0
				else:
					hog_left = hog(roi_left, orientations=4, pixels_per_cell=(4, 4),
			           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
					hog_right = hog(np.flip(roi_right, axis=1), orientations=4, pixels_per_cell=(4, 4),
			           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
				
					similarity = cosine_similarity([hog_left], [hog_right])
					similarity = similarity[0][0]

				if max_similarity < similarity:
					max_similarity = similarity
					best_match = Point((l_x+r_x+small_window_w)//2, c_y)
				c_x += step_w
			# print(f"max similarity: {max_similarity}")
			# print("- - - "*5)
			if max_similarity > initial_similarity:
				central_line_points_processed.append(best_match)
			else:
				central_line_points_processed.append(point)
		i += 1

	return central_line_points_processed


def refine_central_line_avg(central_line_points, threshold) -> list[Point]:
	# Iterates over a list with central line points and if the difference between x position of current and previous point
	# is above threshold corrects the x position of the current point
	print(f"Smooth {len(central_line_points)} central line points ...")
	central_line_points_processed = []
	k = 3
	avg_x = 0
	for point in central_line_points[:k]:
		avg_x += point.x
	avg_x //= k

	for point in central_line_points[:k]:
		diff = point.x - avg_x
		if abs(diff) > threshold and diff > 0:
			point.x -= diff//3
		if abs(diff) > threshold and diff < 0:
			point.x += diff//3
		central_line_points_processed.append(point)

	i = k
	for point in central_line_points[k:-k]:

		if i != k:
			avg_up_x = 0
			for prev_point in central_line_points[i-k:i]:
				avg_up_x += prev_point.x
			avg_up_x //= k
		else:
			avg_up_x = avg_x

		avg_down_x = 0
		for next_point in central_line_points[i:i+k]:
			avg_down_x += next_point.x
		avg_down_x //= k

		diff = max(abs(point.x - avg_down_x), abs(point.x - avg_up_x))

		if abs(diff) > threshold and diff > 0:
			# point.x -= diff//2
			point.x -= int(diff*0.66)
		if abs(diff) > threshold and diff < 0:
			# point.x += diff//2
			point.x += int(diff*0.66)

		central_line_points_processed.append(point)
		i += 1

	return central_line_points_processed
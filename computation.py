import math
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PPoly, splder, splev, splrep


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


def compute_cob_angles(points: list, xb, xe, spline_degree = 5):
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

	smoothing = n - math.sqrt(2 * n)

	# print(f"Number of data points: {n}")

	# ### Define B-spline representation of central line points representing the spine curve
	# k = the degree of the spline; 
	# xb, xe - the interval to fit
	spine_curve = splrep(xs, ys, k=spline_degree, s=smoothing, xb=xb, xe=xe)

	# Define first derivative equation of the spline representation of the spine curve 
	spine_der = splder(spine_curve)

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

	extremums_x = np.insert(arr=extremums_x, obj=0, values=xx[0])
	extremums_x = np.insert(arr=extremums_x, obj=len(extremums_x), values=xx[-1])
	
	extremums_y = splev(extremums_x, spine_curve)
	print(f"Number of extremums: ({extremums_x.shape[0]}, {extremums_y.shape[0]})")
	# print(f"extremums x: \n{extremums_x}")
	# print(f"extremums y: \n{extremums_y}")

	len_extremums = len(extremums_x)
	max_angles = []
	min_x = min(xs)
	max_x = max(xs)
	epsilon = (max_x-min_x)//2
	max_angles = []

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
			print(f"For extremum[{k}] max angle is: {max_angle}")

			# epsilon = epsilon//2
			min_x = extremums_x[k] - epsilon
			max_x = extremums_x[k] + epsilon
			# angle: ( (x1,y1, slope1, c1), (x2,y2, slope2, c2), radian)
			# compute the y-coordinate of the line ends
			y_min_1 = point_eq(min_x, max_angle.t_line_a.slope, max_angle.t_line_a.coefficient)
			y_max_1 = point_eq(max_x, max_angle.t_line_a.slope, max_angle.t_line_a.coefficient)

			y_min_2 = point_eq(min_x, max_angle.t_line_b.slope, max_angle.t_line_b.coefficient)
			y_max_2 = point_eq(max_x, max_angle.t_line_b.slope, max_angle.t_line_b.coefficient)
			

			line_1 = Line(max_angle.point_a, Point(max_x, y_max_1))
			line_2 = Line(max_angle.point_b, Point(min_x, y_min_2))

			# if extremums_y[k] > average_extremums_y:
			# 	line_1 = Line(point_a, Point(max_x, y_max_1))
			# 	line_2 = Line(point_b, Point(min_x, y_min_2))
			# else:
			# 	line_1 = Line(Point(max_x, y_max_1), point_a)
			# 	line_2 = Line(Point(min_x, y_min_2), point_b )

			max_angles.append(CobAngle(apex=Point(extremums_x[k], extremums_y[k]), 
							   		     line_a=line_1, line_b=line_2, measure=rad_to_deg(max_angle.measure)))
			# max_angles_2.append(CobAngle(apex=Point(extremums_x[k], extremums_y[k]), 
			# 				   		     line_a=Line(Point(min_x, y_min_1), Point(max_x, y_max_1)),
			# 				   		     line_b=Line(Point(min_x, y_min_2), Point(max_x, y_max_2)),
			# 				   		     measure=rad_to_deg(max_angle.measure)))
		else:
			np.delete(extremums_x, k)
			np.delete(extremums_y, k)

	return(xx, yy, extremums_x[1:-1], extremums_y[1:-1], max_angles)


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


def find_central_line(spine_crop: np.ndarray) : #-> list[Point]:
	# Slide a window with size (window_w, window_h) over the image with horizontal step step_w
	# and vertical step step_h. At each step calculate the sum of pixel intensities. 
	# At each row identify the window with tha maximum sum and save the central point.
	# Return: List with all central points

	print("Finding central line points...")
	# print(f"spine crop type: {type(spine_crop)}")
	image_h, image_w = spine_crop.shape
	window_w = 60
	window_h = 10
	step_w = 1
	step_h = 5
	# current position of the window
	c_y = c_x = 0

	central_line_points = []
	i = 0
	while c_y < image_h - window_h:
		# (the maximum sum, the central point of the window with the maximum intensity)
		max_sum = (0, Point(0, 0))
		c_x = 0
		while c_x < image_w - window_w:
			roi = spine_crop[c_y:c_y+window_h, c_x:c_x+window_w]
			current_sum  = np.sum(roi)
			if max_sum[0] < current_sum:
				curr_x = c_x + window_w//2
				curr_y = c_y + window_h//2
				max_sum = (current_sum, Point(curr_x, curr_y))
			c_x += step_w
		central_line_points.append(max_sum[1])
		c_y += step_h
	return central_line_points


def refine_central_line(central_line_points, threshold) -> list[Point]:
	# Iterates over a list with central line points and if the difference between x position of current and previous point
	# is above threshold corrects the x position of the current point
	central_line_points_processed = []
	prev_x = central_line_points[0].x
	central_line_points_processed.append(central_line_points[0])
	i = 1
	for point in central_line_points[1:]:
		prev_point = central_line_points[i-1]

		diff = point.x - prev_point.x
		if abs(diff) > threshold:
			point.x = prev_point.x
		# point = (curr_x, point[1])
		central_line_points_processed.append(point)
		i += 1
		# prev_x = curr_x

	return central_line_points_processed


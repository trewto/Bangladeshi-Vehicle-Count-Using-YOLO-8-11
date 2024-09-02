DEFAULT_DETECTION_PATH = 'runs/detect/exp17/labels/Shahbagh_Intersection'
TRAINED_DETECTION_PATH = 'runs/detect/exp24/labels/Shahbagh_Intersection'
FRAME_COUNT = 50160

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

CROSS_LINE = [Point(0.0, .8), Point(.633, .745)]

# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
	if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
		(q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
		return True
	return False

def orientation(p, q, r): 
	# to find the orientation of an ordered triplet (p,q,r)
	# function returns the following values:
	# 0 : Collinear points
	# 1 : Clockwise points
	# 2 : Counterclockwise

	# See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
	# for details of below formula.

	val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
	if (val > 0):
		# Clockwise orientation 
		return 1
	elif (val < 0):
		# Counterclockwise orientation
		return 2
	else:
		# Collinear orientation
		return 0

# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):

	# Find the 4 orientations required for
	# the general and special cases
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if ((o1 != o2) and (o3 != o4)):
		return True

	# Special Cases

	# p1 , q1 and p2 are collinear and p2 lies on segment p1q1
	if ((o1 == 0) and onSegment(p1, p2, q1)):
		return True

	# p1 , q1 and q2 are collinear and q2 lies on segment p1q1
	if ((o2 == 0) and onSegment(p1, q2, q1)):
		return True

	# p2 , q2 and p1 are collinear and p1 lies on segment p2q2
	if ((o3 == 0) and onSegment(p2, p1, q2)):
		return True

	# p2 , q2 and q1 are collinear and q1 lies on segment p2q2
	if ((o4 == 0) and onSegment(p2, q1, q2)):
		return True

	# If none of the cases
	return False

def is_inside_bounding_box(bounding_obj, detected_obj):
	return (2 * abs(bounding_obj[2] - detected_obj[2]) <= bounding_obj[4]) and (2 * abs(bounding_obj[3] - detected_obj[3]) <= bounding_obj[5])

def read_file(frame_path):
	ret = []
	try:
		with open(frame_path) as file:
			for line in file:
				obj = line.rstrip().split()
				for i in range(1, 6):
					try :
						obj[i] = float(obj[i])
					except ValueError :
						pass
				ret.append(obj)
	except:
		return []
	return ret

def read_frame(frame_num):
	default_frame_path = DEFAULT_DETECTION_PATH + '_' + str(frame_num) + '.txt'
	default_frame = read_file(default_frame_path)
	trained_frame_path = TRAINED_DETECTION_PATH + '_' + str(frame_num) + '.txt'
	trained_frame = read_file(trained_frame_path)
	frame = list(trained_frame)
	for obj in default_frame:
		detected_in_trained = False
		for trained_obj in trained_frame:
			if is_inside_bounding_box(trained_obj, obj):
				detected_in_trained = True
				break
		if not detected_in_trained:
			frame.append(obj)
	return frame

def nearest_object(prev_objects, obj):

	if not prev_objects: return None

	min_dist = 1000000.0
	nearest_obj = None
	for prev_obj in prev_objects:
		if prev_obj[0] != obj[0]:
			continue
		dist = ((prev_obj[2] - obj[2]) ** 2) + ((prev_obj[3] - obj[3]) ** 2)
		if dist < min_dist:
			min_dist = dist
			nearest_obj = prev_obj[:]

	if not nearest_obj: return None

	if is_inside_bounding_box(obj, nearest_obj):
		return nearest_obj
	return None


def calculate_volume():

	prev_frame = read_frame(1)
	obj_count = {}
	for frame_num in range(1, FRAME_COUNT):
		now_frame = read_frame(frame_num + 1)
		for obj in now_frame:
			nearest_obj = nearest_object(prev_frame, obj)
			if not nearest_obj:
				continue
			p = Point(obj[2], obj[3])
			q = Point(nearest_obj[2], nearest_obj[3])
			if(doIntersect(CROSS_LINE[0], CROSS_LINE[1], p, q)):
				if obj[0] in obj_count:
					obj_count[obj[0]] += 1
				else:
					obj_count[obj[0]] = 1
		prev_frame = now_frame

	print(obj_count)
		
if __name__ == "__main__":
    calculate_volume()

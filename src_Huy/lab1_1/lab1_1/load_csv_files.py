import csv
import numpy as np
from matplotlib import pyplot as plt
import math


# ###########################################################
# Narcís Palomeras                                          #
# September 2022                                            #
# Universitat de Girona (All rights reserved)               #
#                                                           #
# This project belongs to the Universitat de Girona.        #
# It is forbidden to publish this project or any derivative #
# work in any public repository.                            #
#############################################################


def wrap_angle(angle):
    """ Wraps angle between -pi and pi 

    @type  angle: float or numpy array
    @param angle: angle in radinas

    @rtype:   float or numpy array
    @return:  angle in radinas between -Pi to Pi
    """
    if isinstance(angle, float) or isinstance(angle, int):
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    elif isinstance(angle, np.ndarray):
        return (angle + np.pi) % (2 * np.pi ) - np.pi 
    elif isinstance(angle, list):
        ret = []
        for i in angle:
            ret.append(wrap_angle(i))
        return ret
    else:
        raise NameError('wrap_angle')

   
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    [https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/]
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians
 

def load_encoders(encoders_filepath):
    """
    file: wheel_encoders.csv
    Comments: '%'
    Separator: ','
    Row[0]: time in nanoseconds
    Row[6]: left wheel turns
    Row[7]: right wheel turns
    WARNING! COmparing Forward Kinematics and Odometry it seems that the Y must be negated.
    """
    encoders = []
    # Read wheel encoders form CSV file
    with open(encoders_filepath, 'r') as encoders_file:
        reader = csv.reader(encoders_file, delimiter=',')

        for row in reader:
            if row[0][0] != '%':
                time = int(row[0])/10**9
                wheel_l = float(row[6])
                wheel_r = float(row[7])
                encoders.append((time, wheel_l, wheel_r))

    return np.array(encoders)

def load_laser_scans(scans_filepath):
    """
        Comments: '%'
        Separator: ','
        Row[0]: time in nanoseconds
        Row[4]: angle_min (rad)
        Row[5]: angle_max (rad)
        Row[6]: angle_increment (rad)
        Row[9]: range_min (m)
        Row[10]: range_max (m)
        Row[11-371]: beam_range (m)
    """

    # Read laser_scan_ranges from CSV file
    point_cloud = []
    with open(scans_filepath, 'r') as ls_file:
        reader_ls = csv.reader(ls_file, delimiter=',')
        for row in reader_ls:
            rho = []
            theta = []
            if row[0][0] != '%':
                time = float(row[0])/10**9
                angle_min = float(row[4])
                angle_inc = float(row[6])
                range_min = float(row[9])
                range_max = float(row[10])
                for i, v in enumerate(row[11:]):
                    r = float(v)
                    if r > range_max or r < range_min:
                        r = np.Inf
                    rho.append(r)
                    theta.append(wrap_angle(angle_min + i*angle_inc))
                point_cloud.append((time, rho, theta))
    return point_cloud

    
def load_ground_truth(ground_truth_filepath):
    """ 
    file: ground_truth.csv
    Comments: '%'
    Separator: ','
    Row[0]: time in nanoseconds
    Row[5]: x position
    Row[6]: y position
    Row[10]: z orientation (quaternion)
    """

    stamped_pose = []

    # Read position (x, y) from CSV file
    with open(ground_truth_filepath, 'r') as gt_file:
        reader = csv.reader(gt_file, delimiter=',')
        for row in reader:
            if row[0][0] != '%':
                _, _, theta = euler_from_quaternion(float(row[8]), float(row[9]), float(row[10]), float(row[11]))
                stamped_pose.append([float(row[0])/10**9, float(row[5]), float(row[6]), theta])

    return np.array(stamped_pose)
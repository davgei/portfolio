#!/usr/bin/env python3

"""
This node is designed to take in a circle drawing description and perform
the necessary calculations and commands to draw the circle using the
Crustcrawler platform
"""

from __future__ import print_function
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint
import actionlib
import numpy as np
import rospy


def path_length(path):
    """
    Calculate path length in centimeters

    :param path: List of points
    :returns: Length of path in centimeters
    """
    length = 0.0
    for p1, p0 in zip(path[1:], path):
        length += np.linalg.norm(p1 - p0)
    return length


def inverse_kinematic(position):
    """
    Calculate the inverse kinematic of the Crustcrawler

    :param position: Desired end-point position
    :returns: Three element vector of joint angles
    """
    x, y, z = position
    l1 = 11.0
    l2 = 22.3
    l3 = 17.1
    l4 = 8.0
    l3_mark = np.sqrt(l4 ** 2 + l3 ** 2 + np.sqrt(2) / 2.0 * l4 * l3)
    #phi = np.arccos((l3 ** 2 + l3_mark ** 2 - l4 ** 2)
    #                / (2.0 * l4 * l3_mark))
    s = z - l1
    r = np.sqrt(x ** 2 + y ** 2)
    d = ((x ** 2 + y ** 2 + s ** 2 - l2 ** 2 - l3_mark ** 2)
         / (2. * l2 * l3_mark))

    theta1 = np.arctan2(y, x)
    theta3 = np.arctan2(-np.sqrt(1. - d ** 2), d)
    theta2 = np.arctan2(s, r) - np.arctan2(l3_mark * np.sin(theta3),
                                           l2 + l3_mark * np.cos(theta3))
    return np.array([theta1, theta2 * -1. + np.pi / 2., theta3 * -1.])


def create_trajectory_point(position, seconds):
    """
    Create a JointTrajectoryPoint

    :param position: Joint positions
    :param seconds: Time from start in seconds
    :returns: JointTrajectoryPoint
    """
    point = JointTrajectoryPoint()
    point.positions.extend(position)
    point.time_from_start = rospy.Duration(seconds)
    return point

def generate_path(origin, radius, num):
    """
    Generate path in 3D space of where to draw circle

    :param origin: 3D point of circle origin
    :param radius: Radius of circle in centimeters
    :param num: Number of points in circle
    :returns: List of points to draw
    """
    angles = np.linspace(0, 2*np.pi, num, endpoint=False)
    path = []
    for a in angles:
        x = origin[0] + radius*np.cos(a)
        y = origin[1] + radius*np.sin(a)
        z = origin[2]              
        path.append(np.array([x, y, z]))
    return path


def generate_movement(path):
    """
    Generate Crustcrawler arm movement through a message

    :param path: List of points to draw
    :returns: FollowJointTrajectoryGoal describing the arm movement
    """
    movement = FollowJointTrajectoryGoal()

    # Navn og toleranser
    movement.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3']
    tol = [JointTolerance('joint_'+str(i+1), 0.1, 0., 0.) for i in range(3)]
    movement.goal_tolerance.extend(tol)
    movement.goal_time_tolerance = rospy.Duration(0.5)

    time = 4.0
    # Startstilling
    movement.trajectory.points.append(
        create_trajectory_point([0., 0., np.pi/2.], time))

    # først til sirkelførste punkt (bruk inverse_kinematic)
    time += 4.0
    movement.trajectory.points.append(
        create_trajectory_point(inverse_kinematic(path[0]), time))

    # beregn tid per segment ~2 cm/s
    length = path_length(path)
    dt = (length / 2.0) / len(path)

    for p in path[1:]:
        time += dt
        movement.trajectory.points.append(
            create_trajectory_point(inverse_kinematic(p), time))

    # tilbake til hvile
    time += 4.0
    movement.trajectory.points.append(
        create_trajectory_point([0., 0, np.pi/2.], time))

    return movement


def draw_circle(origin, radius, num):
    """
    Draw circle using Crustcrawler

    :param origin: 3D point of circle origin
    :param radius: Radius of circle in centimeters
    :param num: Number of points in circle
    """
    client = actionlib.SimpleActionClient(
        '/crustcrawler/controller/follow_joint_trajectory',
        FollowJointTrajectoryAction)

    path  = generate_path(origin, radius, num)
    goal  = generate_movement(path)

    client.wait_for_server()
    client.send_goal(goal)
    client.wait_for_result()

    res = client.get_result()
    if not res.error_code:
        rospy.loginfo("Crustcrawler done!")
    else:
        rospy.logwarn("Crustcrawler failed: %s (%s)",
                      res.error_string, res.error_code)
    return res.error_code


if __name__ == '__main__':
    import argparse
    import sys
    # Create command line parser and add options:
    parser = argparse.ArgumentParser(
            description="CrustCrawler circle drawer TM(C), patent pending!")
    parser.add_argument(
            '--origin', '-o', type=float, nargs=3,
            metavar=('x', 'y', 'z'), required=True,
            help="Origin of the board")
    parser.add_argument(
            '--radius', '-r', type=float, default=5.0,
            metavar='cm', help="The radius of the circle to draw")
    parser.add_argument(
            '--num_points', '-n', type=int,
            default=4, metavar='num',
            help="Number of points to use when drawing the circle")
    args = parser.parse_args()
    # Ensure points are NumPy arrays
    args.origin = np.array(args.origin)
    orient = np.array([0, 1., 0])
    # Ensure that arguments are within legal limits:
    if 3 >= args.num_points <= 101:
        sys.exit("Number of points must be in range [3, 101] was: {:d}"
                 .format(args.num_points))
    max_dist = np.linalg.norm(args.origin)
    if np.abs(max_dist - args.radius) < 20.0:
        sys.exit("Circle is to close to the robot or outside the workspace! Minimum: 20cm, was: {:.2f}"
                 .format(max_dist - args.radius))
    # Create ROS node
    rospy.init_node('circle_drawer', anonymous=True)
    # Call function to draw circle
    try:
        sys.exit(draw_circle(args.origin, args.radius, args.num_points))
    except rospy.ROSInterruptException:
        sys.exit("Program aborted during circle drawing")

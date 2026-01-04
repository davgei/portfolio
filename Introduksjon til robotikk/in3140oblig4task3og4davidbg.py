#!/usr/bin/env python3

"""
Implementation of a PID controller.

Assignment 4, in3140
"""
import rospy
import math


class PID(object):
    def __init__(self):
        # Proportional constant
        self.p = 0.0
        # Integral constant
        self.i = 0.0
        # Integral accumulation variable
        self.integral = 0.0
        # Derivative constant
        self.d = 0.0
        # Non-linear constant
        self.c = 0.0
        # Position error
        self.error = 0.0

    def __call__(self, desired_theta, current_theta, velocity_theta, dt):
        """
        Perform PID control step

        :param desired_theta: Desired set-point in radians
        :param current_theta: Current joint angle in radians
        :param velocity_theta: Current joint angle velocity in radians/second
        :param dt: Time since last call in seconds
        :returns: Effort for joint
        """
        # TODO: Change which line is commented according to which part
        # you are testing in your code.
        # return self.P_ctrl(desired_theta, current_theta, dt)
        #return self.PD_ctrl(desired_theta, current_theta, velocity_theta, dt)
        #return self.PID_ctrl(desired_theta, current_theta, velocity_theta, dt)
        return self.PI_ctrl(desired_theta, current_theta, dt)
        # return self.PIDD_ctrl(desired_theta, current_theta, velocity_theta, dt)

    def P_ctrl(self, desired_theta, current_theta, dt):
        """
        Calculate proportional control

        :param desired_theta: Desired set-point in radians
        :param current_theta: Current joint angle in radians
        :param dt: Time since last call in seconds
        :returns: Effort of joint
        """
        # TODO: Implement!
        # TIP: Use 'rospy.loginfo' to print output in ROS
        return 0.0

    def PD_ctrl(self, desired_theta, current_theta, velocity_theta, dt):
        """
        Calculate Proportional-Derivative control

        :param desired_theta: Desired set-point in radians
        :param current_theta: Current joint angle in radians
        :param velocity_theta: Current joint angle velocity in radians/second
        :param dt: Time since last call in seconds
        :returns: Effort for joint
        """
        # TODO: Implement!
        self.error  = desired_theta - current_theta
        effort = self.p*self.error - self.d*velocity_theta
        # printer ut infor om pd
        rospy.loginfo("PD ⇒ err=%.3f  vel=%.3f  effort=%.3f", 
                      self.error, velocity_theta, effort)
        return effort

    def PI_ctrl(self, desired_theta, current_theta, dt):
        """
        Calculate PID control

        :param desired_theta: Desired set-point in radians
        :param current_theta: Current joint angle in radians
        :param velocity_theta: Current joint angle velocity in radians/second
        :param dt: Time since last call in seconds
        :returns: Effort for joint
        """
        # TODO: Implement!
        # TIP: Use 'rospy.loginfo' to print output in ROS
        self.error  = desired_theta - current_theta
        self.integral += self.error*dt

        effort = self.p*(self.error+ (self.integral/self.p)*self.i)
        return effort

    def PID_ctrl(self, desired_theta, current_theta, velocity_theta, dt):
        """
        Calculate PID control

        :param desired_theta: Desired set-point in radians
        :param current_theta: Current joint angle in radians
        :param velocity_theta: Current joint angle velocity in radians/second
        :param dt: Time since last call in seconds
        :returns: Effort for joint
        """
        # TODO: Implement!
        # TIP: Use 'rospy.loginfo' to print output in ROS
        self.error  = desired_theta - current_theta
        self.integral += self.error*dt

        effort = self.p*(self.error+ (self.integral/self.p)*self.i)
        return effort

    def PIDD_ctrl(self, desired_theta, current_theta, velocity_theta, dt):
        """
        Calculate non-linear PID control

        :param desired_theta: Desired set-point in radians
        :param current_theta: Current joint angle in radians
        :param velocity_theta: Current joint angle velocity in radians/second
        :param dt: Time since last call in seconds
        :returns: Effort for joint
        """
        # TODO: Implement!
        # TIP: Use 'rospy.loginfo' to print output in ROS
        return 0.0

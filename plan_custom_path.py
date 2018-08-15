import sys
import copy
import rospy
import math
import pickle
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
from moveit_msgs.msg._RobotState import RobotState
from moveit_msgs.msg._RobotTrajectory import RobotTrajectory
from moveit_msgs.srv import GetPositionFK, GetStateValidity, GetStateValidityRequest, GetStateValidityResponse
from trajectory_msgs.msg._JointTrajectoryPoint import JointTrajectoryPoint
from std_msgs.msg import Header

import actionlib
import control_msgs.msg

from math import cos, sin, pi

if __name__ == '__main__':
    rospy.init_node('rospy_log_loader',
                    anonymous=True)
    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander("widowx_arm")

    rs = RobotState()
    rs.joint_state.name = joint_names
    display_trajectory_publisher = rospy.Publisher(
                                        '/move_group/display_planned_path',
                                        moveit_msgs.msg.DisplayTrajectory,
                                        queue_size=10)
    # close_plan = gripper_group.plan((0.031, 0.0))

    master_plan = RobotTrajectory()
    states = pickle.load(open('open_listened_states.pkl', 'rb+'))
    master_plan.joint_trajectory.joint_names = joint_names
    rs.joint_state = (states[0][0], states[0][1], states[0][2], states[0][3], 0.0)
    for state in states:
        point = JointTrajectoryPoint()
        point.positions = (state[0], state[1], state[2], state[3], 0.0)
        master_plan.joint_trajectory.points.append(point)

    print("============ Waiting while plan is loaded...")
    rospy.sleep(5)
    print("============ Publishing Plan")
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(master_plan)
    display_trajectory_publisher.publish(display_trajectory)

    print("============ Waiting while plan is published...")
    rospy.sleep(5)
    print('Published')

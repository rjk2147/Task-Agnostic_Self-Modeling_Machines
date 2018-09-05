import sys
import copy
import rospy
import math
import pickle
import time
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
from moveit_msgs.msg._RobotState import RobotState
from moveit_msgs.msg._RobotTrajectory import RobotTrajectory
from moveit_msgs.srv import GetPositionFK, GetStateValidity, GetStateValidityRequest, GetStateValidityResponse
from trajectory_msgs.msg._JointTrajectoryPoint import JointTrajectoryPoint
from std_msgs.msg import Header

if __name__ == '__main__':
    rospy.init_node('rospy_listener',
                anonymous=True)
    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander("widowx_arm")
    gripper_group = moveit_commander.MoveGroupCommander("widowx_gripper")

    states = []
    next_update = 5
    true_start = time.time()
    print('Listening...')
    while time.time()-true_start < 200:
        start = time.time()
        state = robot.get_current_state().joint_state.position
        if time.time()-true_start > next_update:
            print(str(time.time()-true_start)+': '+str(state))
            next_update+=5
        states.append(state)
        while time.time()-start < 0.05: pass
    pickle.dump(states, open('real_closed_listened_states.pkl','wb+'))
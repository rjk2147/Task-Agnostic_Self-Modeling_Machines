import gym
import math
import numpy as np
from gym import spaces
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

from test_planners.control_widowx import calc_end_effector_pos, clip_joints

# TODO migrate ROS interface to python3
class WidowxROS(gym.Env):
    def __init__(self):
        print("============ Starting setup")
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface_tutorial',
                        anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("widowx_arm")
        self.end = self.group.get_end_effector_link()
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=10)

        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']

        rospy.wait_for_service('compute_fk')
        try:
            self.moveit_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
        except rospy.ServiceException as e:
            rospy.logerror("Service call failed: %s" % e)
        # self.header = Header(0, rospy.Time.now(), "/world")
        self.rs = RobotState()
        self.rs.joint_state.name = joint_names

        self.stateValidator = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        self.gsvr = GetStateValidityRequest()
        self.gsvr.group_name = 'widowx_arm'


        # last 3 are gathered from data, more precision needed there
        bounds = np.array([2.617, 1.571, 1.571, 1.745, 0.37, 0.37, 0.51])
        # bounds = np.array([2.617, 1.571, 1.571, 1.745, 2.617, 0.37, 0.37, 0.51])
        self.action_space = spaces.Box(-math.pi/16.0, math.pi/16.0, shape=(4,))
        self.observation_space = spaces.Box(low=-bounds, high=bounds)

        self.max_iter = 100

    def reset(self):
        collision = True
        self.iteration = 0

        self.angles = np.array(self.group.get_random_joint_values())
        self.angles = self.angles[:-1]
        while collision:
            self.angles = np.array(self.group.get_random_joint_values())
            self.angles = self.angles[:-1]
            self.rs.joint_state.position = np.concatenate([self.angles, np.array([0])], axis=0)
            self.gsvr.robot_state = self.rs
            collision = not self.stateValidator.call(self.gsvr).valid
        self.rs.joint_state.position = np.concatenate([self.angles, np.array([0])], axis=0)
        # end_effector = self.moveit_fk(self.header, ['wrist_2_link'], self.rs).pose_stamped[0].pose


        self.target = calc_end_effector_pos(self.angles)
        # self.target = get_end_effector_pos(self.moveit_fk, self.rs)


        self.angles = np.array(self.group.get_random_joint_values())
        self.angles = self.angles[:-1]
        while collision:
            self.angles = np.array(self.group.get_random_joint_values())
            self.angles = self.angles[:-1]
            self.rs.joint_state.position = np.concatenate([self.angles, np.array([0])], axis=0)
            self.gsvr.robot_state = self.rs
            collision = not self.stateValidator.call(self.gsvr).valid
        self.rs.joint_state.position = np.concatenate([self.angles, np.array([0])], axis=0)
        # end_effector = self.moveit_fk(self.header, ['wrist_2_link'], self.rs).pose_stamped[0].pose

        # self.position = get_end_effector_pos(self.moveit_fk, self.rs)
        self.position = calc_end_effector_pos(self.angles)

        self.state = np.concatenate([self.angles, self.position], axis=0)

        self.d = float(np.linalg.norm(self.position - self.target))


        return self.state

    def step(self, action):
        self.iteration += 1
        # New Joint State
        test_angle = clip_joints(self.angles+action)

        # Using ROS to test forward kinematics and prevent collision
        self.rs.joint_state.position = np.concatenate([test_angle, np.array([0])], axis=0)
        self.gsvr.robot_state = self.rs
        collision = not self.stateValidator.call(self.gsvr).valid

        # Only updating if there is no collision and thus valid
        # new_state = np.zeros_like(state)+state
        if not collision:
            # self.position = get_end_effector_pos(self.moveit_fk, self.rs)
            self.position = calc_end_effector_pos(self.angles)

            # self.orientation = np.array([end_effector.orientation.w, end_effector.orientation.x, end_effector.orientation.y,
            #                         end_effector.orientation.z])

            self.angles = np.zeros_like(test_angle)+test_angle
            self.state = np.concatenate([self.angles, self.position], axis=0)
        # else:
        #     print('Step '+str(episode_step)+' Collision!')
        # angles_list.append(np.zeros_like(angles)+angles)
        # episode.append([state, action*max_action, 0.0, new_state, (episode_step==99), episode_step])
        self.done = (self.iteration>=99)

        new_d = float(np.linalg.norm(self.position - self.target))
        if self.iteration == 1:
            r = -new_d
        else:
            r = self.d - new_d
        return self.state, r, self.done, {}
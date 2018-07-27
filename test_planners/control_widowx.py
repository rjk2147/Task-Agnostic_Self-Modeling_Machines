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

def clip_joints(val):
    bounds = [2.617, 1.571, 1.571, 1.745, 2.617]
    for j in range(len(bounds)):
        val[j] = max(-bounds[j], min(bounds[j], val[j]))
    return val


print("============ Starting setup")
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial',
                anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("widowx_arm")
end = group.get_end_effector_link()
display_trajectory_publisher = rospy.Publisher(
                                    '/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory,
                                    queue_size=10)

joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5']


rospy.wait_for_service('compute_fk')
try:
  moveit_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
except rospy.ServiceException as e:
  rospy.logerror("Service call failed: %s"%e)
header = Header(0,rospy.Time.now(),"/world")
rs = RobotState()
rs.joint_state.name = joint_names

stateValidator = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
gsvr = GetStateValidityRequest()
gsvr.group_name = 'widowx_arm'

# Data generation params
max_action = math.pi/16.0

# Training data generation
nb_episodes = 1000

# Reset State
train = []
for n in range(nb_episodes):
    # First random angle loop double checking no self-collision
    collision = True
    angles = np.array(group.get_random_joint_values())
    while collision:
        angles = np.array(group.get_random_joint_values())
        rs.joint_state.position = angles
        gsvr.robot_state = rs
        collision = not stateValidator.call(gsvr).valid

    angles_list = [np.array([angles[0], angles[1], angles[2], angles[3], angles[4]])]
    episode = []
    rs.joint_state.position = angles
    end_effector = moveit_fk(header, ['wrist_2_link'], rs).pose_stamped[0].pose
    position = np.array([end_effector.position.x, end_effector.position.y, end_effector.position.z])
    state = np.concatenate([angles, position], axis=0)
    for episode_step in range(100): # Make 100000 for real training data
        # New Joint State
        action = np.random.uniform(-1, 1, len(angles))
        test_angle = clip_joints(angles+action*max_action)

        # Using ROS to test forward kinematics and prevent collision
        rs.joint_state.position = test_angle
        gsvr.robot_state = rs
        end_effector = moveit_fk(header, ['wrist_2_link'], rs).pose_stamped[0].pose
        collision = not stateValidator.call(gsvr).valid

        # Only updating if there is no collision and thus valid
        new_state = np.zeros_like(state)+state
        if not collision:
            position = np.array([end_effector.position.x, end_effector.position.y, end_effector.position.z])
            orientation = np.array([end_effector.orientation.w, end_effector.orientation.x, end_effector.orientation.y,
                                    end_effector.orientation.z])

            new_state = np.concatenate([angles, position], axis=0)
            angles = np.zeros_like(test_angle)+test_angle
        # else:
        #     print('Step '+str(episode_step)+' Collision!')
        angles_list.append(np.zeros_like(angles)+angles)
        episode.append([state, action*max_action, 0.0, new_state, (episode_step==99), episode_step])
    print('Episode '+str(n)+' complete')
    train.extend(episode)

print(len(train))
pickle.dump(train, open('widowx_train.pkl', 'wb+'))

# Linear Interpolation along angles for smoother/slower traveling (less wear on motors)
int_len = 10
for i in range(len(angles_list)-1):
    diff = angles_list[i*int_len+1]-angles_list[i*int_len]
    for j in range(1,int_len):
        val = angles_list[i*int_len]+float(j)*diff/float(int_len)
        angles_list.insert(i*int_len+j, val)

# Converting data into ROS path for execution
# Plan Setup
plan = RobotTrajectory()
plan.joint_trajectory.joint_names = joint_names
point = JointTrajectoryPoint()
point.positions = (angles[0], angles[1], angles[2], angles[3], angles[4])
plan.joint_trajectory.points.append(point)

# Populating Plan
i = 0
for angles in angles_list:
    point = JointTrajectoryPoint()
    point.positions = (angles[0], angles[1], angles[2], angles[3], angles[4])
    plan.joint_trajectory.points.append(point)
    i += 1

# Publishing to ROS
print("============ Waiting while plan is loaded...")
rospy.sleep(5)

print("============ Publishing Plan")
display_trajectory = moveit_msgs.msg.DisplayTrajectory()
display_trajectory.trajectory_start = robot.get_current_state()
display_trajectory.trajectory.append(plan)
display_trajectory_publisher.publish(display_trajectory)

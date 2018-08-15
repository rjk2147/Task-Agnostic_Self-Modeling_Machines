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
from math import sin, cos, pi

def clip_joints(val):
    bounds = [2.617, 1.571, 1.571, 1.745]
    for j in range(len(bounds)):
        val[j] = max(-bounds[j], min(bounds[j], val[j]))
    return val

# def get_end_effector_pos(moveit_fk, rs):
#     header = Header(0,rospy.Time.now(),"/world")
#     grip1 = moveit_fk(header, ['gripper_1_link'], rs).pose_stamped[0].pose
#     grip2 = moveit_fk(header, ['gripper_2_link'], rs).pose_stamped[0].pose
#     p1 = np.array([grip1.position.x, grip1.position.y, grip1.position.z])
#     p2 = np.array([grip2.position.x, grip2.position.y, grip2.position.z])
#     position = (p1+p2)/2
#     return position

def translate(p, obj):
    t = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [p[0], p[1], p[2], 1]]
                )
    return np.dot(t, obj)

def rotate(theta, obj):
    # r = [math.radians(theta[0]), math.radians(theta[1]), math.radians(theta[2])]
    # r = np.radians(theta)
    r = theta
    Rx = np.array([[1, 0, 0, 0],
                  [0, cos(r[0]), -sin(r[0]), 0],
                  [0, sin(r[0]), cos(r[0]), 0],
                  [0, 0, 0, 1]]
                )
    Ry = np.array([[cos(r[1]), 0, sin(r[1]), 0],
                  [0, 1, 0, 0],
                  [-sin(r[1]), 0, cos(r[1]), 0],
                  [0, 0, 0, 1]]
                )
    Rz = np.array([[cos(r[2]), -sin(r[2]), 0, 0],
                  [sin(r[2]), cos(r[2]), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
                )
    R = np.dot(Rx, Ry)
    R = np.dot(R, Rz)
    return np.dot(R, obj)

def calc_end_effector_pos(angles):
    return calc_default_end_effector_pos(angles)
    # return calc_deformed_end_effector_pos(angles)

def calc_default_end_effector_pos(angles):
    obj = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
                    )

    # Base Transform
    obj = translate([0.0,0.0,0.125], obj)

    # Bicep FK
    obj = rotate([0, -angles[1], -angles[0]], obj)
    obj = translate([0.04825, 0, 0.14203], obj)

    # Forearm FK
    obj = rotate([0, -pi/2-angles[2], 0], obj)
    obj = translate([0, 0, 0.14203], obj)

    # Wrist FK
    obj = rotate([0, -angles[3], 0], obj)
    obj = translate([0, 0, 0.11450], obj)

    # End Effector Position
    return obj[3][:3]


# with new part angled downward
def calc_deformed_end_effector_pos(angles):
    obj = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
                    )

    # Base Transform
    obj = translate([0.0,0.0,0.125], obj)

    # Bicep FK
    obj = rotate([0, -angles[1], -angles[0]], obj)
    obj = translate([0.04825, 0, 0.14203], obj)

    # print(obj[3][:3])
    # Forearm FK
    obj = rotate([0, -pi/2-angles[2], 0], obj)
    obj = translate([0, 0, 0.073], obj)
    obj = rotate([0, -pi/6, 0], obj)
    obj = translate([0, 0, 0.074], obj)
    # print(obj[3][:3])

    # Wrist FK
    obj = rotate([0, -angles[3], 0], obj)
    obj = translate([0, 0, 0.11450], obj)

    # End Effector Position
    return obj[3][:3]
if __name__ == '__main__':
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

    header = Header(0, rospy.Time.now(), "/world")

    rospy.wait_for_service('compute_fk')
    try:
      moveit_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
    except rospy.ServiceException as e:
      rospy.logerror("Service call failed: %s"%e)
    # header = Header(0,rospy.Time.now(),"/world")
    rs = RobotState()
    rs.joint_state.name = joint_names

    stateValidator = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
    gsvr = GetStateValidityRequest()
    gsvr.group_name = 'widowx_arm'


    # Training data generation
    nb_episodes = 1000
    episode_len = 100

    # Data generation params
    max_action = nb_episodes*(math.pi/16.0)/1000


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
        angles = angles[:-1]
        angles_list = [np.array([angles[0], angles[1], angles[2], angles[3]])]
        episode = []
        rs.joint_state.position = angles
        # end_effector = moveit_fk(header, ['wrist_2_link'], rs).pose_stamped[0].pose
        # position = np.array([end_effector.position.x, end_effector.position.y, end_effector.position.z])

        # position = get_end_effector_pos(moveit_fk, rs)
        position = calc_end_effector_pos(angles)

        state = np.concatenate([angles, position], axis=0)
        for episode_step in range(episode_len): # Make 100000 for real training data
            # New Joint State
            action = np.random.uniform(-1, 1, len(angles))
            test_angle = clip_joints(angles+action*max_action)

            # Using ROS to test forward kinematics and prevent collision
            rs.joint_state.position = np.concatenate([test_angle, np.array([0])], axis=0)
            gsvr.robot_state = rs
            # end_effector = moveit_fk(header, ['wrist_2_link'], rs).pose_stamped[0].pose
            collision = not stateValidator.call(gsvr).valid

            # Only updating if there is no collision and thus valid
            new_state = np.zeros_like(state)+state
            if not collision:
                # position = get_end_effector_pos(moveit_fk, rs)
                position = calc_end_effector_pos(test_angle)

                # position = np.array([end_effector.position.x, end_effector.position.y, end_effector.position.z])
                # orientation = np.array([end_effector.orientation.w, end_effector.orientation.x, end_effector.orientation.y,
                #                         end_effector.orientation.z])
                angles = np.zeros_like(test_angle)+test_angle

                new_state = np.concatenate([angles, position], axis=0)
            # else:
            #     print('Step '+str(episode_step)+' Collision!')
            angles_list.append(np.zeros_like(angles)+angles)
            episode.append([state, action*max_action, 0.0, new_state, (episode_step==99), episode_step])
            state = np.zeros_like(new_state)+new_state
        print('Episode '+str(n)+' complete')
        train.extend(episode)

    print(len(train))
    print(len(angles_list))
    file_name = 'widowx_train_10hz_100K_deformed.pkl'
    pickle.dump(train, open(file_name, 'wb+'))
    print('Dumped to '+str(file_name))

    # # Linear Interpolation along angles for smoother/slower traveling (less wear on motors)
    # # int_len = 10
    # # for i in range(len(angles_list)-1):
    # #     diff = angles_list[i*int_len+1]-angles_list[i*int_len]
    # #     for j in range(1,int_len):
    # #         val = angles_list[i*int_len]+float(j)*diff/float(int_len)
    # #         angles_list.insert(i*int_len+j, val)
    #
    # Converting data into ROS path for execution
    # Plan Setup
    # plan = RobotTrajectory()
    # plan.joint_trajectory.joint_names = joint_names
    # point = JointTrajectoryPoint()
    # point.positions = (angles[0], angles[1], angles[2], angles[3], 0.0)
    # plan.joint_trajectory.points.append(point)
    #
    # # Populating Plan
    # for i in range(100):
    #     state = train[i][0]
    #     angles = state[:-3]
    #     pos = state[-3:]
    #     point = JointTrajectoryPoint()
    #     point.positions = (angles[0], angles[1], angles[2], angles[3], 0.0)
    #     plan.joint_trajectory.points.append(point)
    #
    #     print(i)
    #     print(pos)
    #     print('')
    #
    # # Publishing to ROS
    # print("============ Waiting while plan is loaded...")
    # rospy.sleep(5)
    #
    # print("============ Publishing Plan")
    # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    # display_trajectory.trajectory_start = robot.get_current_state()
    # display_trajectory.trajectory.append(plan)
    # display_trajectory_publisher.publish(display_trajectory)

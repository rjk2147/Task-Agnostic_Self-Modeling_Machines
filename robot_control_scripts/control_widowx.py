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
    # return calc_default_end_effector_pos(angles)
    return calc_deformed_end_effector_pos(angles)

def calc_wrist_pos(angles):
    # return calc_default_end_effector_pos(angles)
    return calc_deformed_wrist_pos(angles)

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
    obj = translate([0, 0, 0.06], obj)
    obj = rotate([0, -pi/6, 0], obj)
    obj = translate([0, 0, 0.075], obj)
    obj = translate([0, 0, 0.06], obj)
    # print(obj[3][:3])

    # Wrist FK
    obj = rotate([0, -angles[3], 0], obj)
    obj = translate([0, 0, 0.11450], obj)

    # End Effector Position
    return obj[3][:3]

# with new part angled downward
def calc_deformed_wrist_pos(angles):
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
    obj = translate([0, 0, 0.06], obj)
    obj = rotate([0, -pi/6, 0], obj)
    obj = translate([0, 0, 0.075], obj)
    obj = translate([0, 0, 0.06], obj)

    # Wrist Position
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
    nb_episodes = 100
    episode_len = 100

    # Data generation params
    max_action = math.pi/16.0

    def get_real_angles(angles):
        tmp_angles = np.concatenate([angles, np.array([0])], axis=0)
        group.execute(group.plan(tmp_angles), wait=True)
        real_angles = robot.get_current_state().joint_state.position
        return real_angles[:-1]

    def is_collision(sim_angles):
        rs.joint_state.position = sim_angles
        gsvr.robot_state = rs
        collision = not stateValidator.call(gsvr).valid

        wrist_pos = calc_wrist_pos(sim_angles)
        end_effector_pos = calc_end_effector_pos(sim_angles)

        if wrist_pos[-1] < 0.07: return True
        if end_effector_pos[-1] < 0.05: return True
        wrist_center_dist = np.linalg.norm(wrist_pos[:-1])
        end_effector_center_dist = np.linalg.norm(end_effector_pos[:-1])
        if wrist_center_dist < 0.08 and wrist_pos[-1] < 0.15: return True
        if end_effector_center_dist < 0.08 and end_effector_pos[-1] < 0.15: return True
        return collision

    # Reset State
    sim_train = []
    real_train = []
    import time
    start = time.time()
    batch_start = time.time()
    for n in range(nb_episodes):
        # First random angle loop double checking no self-collision
        collision = True
        sim_angles = np.array(group.get_random_joint_values())[:-1]
        while collision:
            sim_angles = np.array(group.get_random_joint_values())[:-1]
            collision = is_collision(sim_angles)
        # sim_angles = sim_angles[:-1]
        # print(sim_angles)
        real_angles = get_real_angles(sim_angles)

        # angles_list = [np.array([angles[0], angles[1], angles[2], angles[3]])]
        sim_episode = []
        real_episode = []
        rs.joint_state.position = sim_angles
        # end_effector = moveit_fk(header, ['wrist_2_link'], rs).pose_stamped[0].pose
        # position = np.array([end_effector.position.x, end_effector.position.y, end_effector.position.z])


        # Reseting
        # position = get_end_effector_pos(moveit_fk, rs)
        sim_position = calc_end_effector_pos(sim_angles)
        real_position = calc_end_effector_pos(real_angles)


        sim_state = np.concatenate([sim_angles, sim_position], axis=0)
        real_state = np.concatenate([real_angles, real_position], axis=0)

        for episode_step in range(episode_len): # Make 100000 for real training data
            # New Joint State
            action = np.random.uniform(-1, 1, len(sim_angles))
            test_sim_angle = clip_joints(sim_angles+action*max_action)
            # test_real_angle = clip_joints(sim_angles+action*max_action)

            # Using ROS to test forward kinematics and prevent collision
            rs.joint_state.position = np.concatenate([test_sim_angle, np.array([0])], axis=0)
            collision = is_collision(sim_angles)

            # Only updating if there is no collision and thus valid
            new_sim_state = np.zeros_like(sim_state)+sim_state
            new_real_state = np.zeros_like(real_state)+real_state
            if not collision:

                # Taking Step
                # position = get_end_effector_pos(moveit_fk, rs)

                sim_angles = np.zeros_like(test_sim_angle)+test_sim_angle
                real_angles = get_real_angles(sim_angles)

                sim_position = calc_end_effector_pos(sim_angles)
                real_position = calc_end_effector_pos(real_angles)

                new_sim_state = np.concatenate([sim_angles, sim_position], axis=0)
                new_real_state = np.concatenate([real_angles, real_position], axis=0)

            # Storing data
            sim_episode.append([sim_state, action*max_action, 0.0, new_sim_state, (episode_step==99), episode_step])
            sim_state = np.zeros_like(new_sim_state)+new_sim_state

            real_episode.append([real_state, action*max_action, 0.0, new_real_state, (episode_step==99), episode_step])
            real_state = np.zeros_like(new_real_state)+new_real_state

        print('Episode '+str(n)+' complete')
        sim_train.extend(sim_episode)
        real_train.extend(real_episode)

        if n%10 == 9:
            print(len(sim_train))
            end = time.time()
            print('Batch Duration: '+str(end-batch_start))
            print('Total Duration: '+str(end-start))
            # print(len(angles_list))
            file_name = 'deformed_widowx_train_10hz_100K.pkl'
            pickle.dump(sim_train, open('sim_'+file_name, 'wb+'))
            pickle.dump(real_train, open('real_'+file_name, 'wb+'))
            print('Dumped to '+str(file_name))
            print('Resting for 100s...')
            while time.time()-end < 100: pass
            print('Rest Completed')
            batch_start = time.time()

    print(len(sim_train))
    end = time.time()
    # print('Batch Duration: '+str(end-batch_start))
    print('Total Duration: '+str(end-start))
    # print(len(angles_list))
    file_name = 'deformed_widowx_train_10hz_100K.pkl'
    pickle.dump(sim_train, open('sim_'+file_name, 'wb+'))
    pickle.dump(real_train, open('real_'+file_name, 'wb+'))
    print('Dumped to '+str(file_name))
    batch_start = time.time()

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

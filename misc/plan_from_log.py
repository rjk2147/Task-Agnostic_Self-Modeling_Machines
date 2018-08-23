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


def read_until(file, char):
    ret_str = ''
    cursor = file.read(1)
    while cursor != char:
        ret_str += cursor
        cursor = file.read(1)
    return ret_str

def get_angles_from_log(log_path):
    from os import listdir
    from os.path import isfile, join
    # files = [f for f in listdir(log_path) if isfile(join(log_path, f)) and '_log.txt' in f]

    log_file = open(log_path)
    start_at = 0
    nb_episodes = 100
    max_steps = 100

    start = log_file.readline()
    idx = int(start.split()[1])

    all_real_angles = []
    all_pred_angles = []

    while idx < nb_episodes:
        print('Episode '+str(idx))
        episode_pred_states = []
        episode_real_states = []

        read_until(log_file, '[')
        target = np.array(read_until(log_file, ']').split(), dtype='float64')
        read_until(log_file, '[')

        pred_state = np.array(read_until(log_file, ']').split(), dtype='float64')
        log_file.read(1)
        pred_d = float(log_file.readline().split()[2])

        read_until(log_file, '[')
        real_state = np.array(read_until(log_file, ']').split(), dtype='float64')
        log_file.read(1)
        real_d = float(log_file.readline().split()[2])
        drift = float(log_file.readline().split()[1])
        act_idx = 0

        episode_pred_states.append(pred_state)
        episode_real_states.append(real_state)

        while True:
            test_char = log_file.read(1)
            if test_char == 'E':
                test_str = test_char+log_file.readline()
                if test_str[1] == 'p':
                    break
                else:
                    print('Error Reading Log')
                    exit()

            # Get Action
            action_str = read_until(log_file, ']').split()
            act_idx = int(action_str[1][:-1])
            action_str[2] = action_str[2][1:]
            action = np.array(action_str[2:])
            read_until(log_file, ':')
            log_file.read(1)
            reward = float(read_until(log_file, 'P'))
            read_until(log_file, '[')

            # Get Pred State
            pred_state = np.array(read_until(log_file, ']').split(), dtype='float64')

            # Pred D
            read_until(log_file, ':')
            pred_d = float(log_file.readline().strip())

            read_until(log_file, '[')
            # Get Real State
            real_state = np.array(read_until(log_file, ']').split(), dtype='float64')

            # Real D
            read_until(log_file, ':')
            real_d = float(log_file.readline().strip())
            drift = float(log_file.readline().split()[1])


            episode_pred_states.append(pred_state)
            episode_real_states.append(real_state)

        # if idx >= start_at:
        #     make_gif(episode_pred_states, episode_real_states, target=target, name='episode_'+str(idx), path=log_path, successes=True)
        #     make_gif(episode_pred_states, episode_real_states, target=target, name='episode_'+str(idx), path=log_path, successes=False)
        if len(episode_pred_states) <= 100:
            all_pred_angles.append(episode_pred_states)
            all_real_angles.append(episode_real_states)

        log_file.readline()
        start = log_file.readline()
        try:
            idx = int(start.split()[1])
        except:
            print
            print('Finished '+log_path)
            return all_real_angles, all_pred_angles
    return all_real_angles, all_pred_angles


def get_end_effector_pos(moveit_fk, rs):
    header = Header(0,rospy.Time.now(),"/world")
    grip1 = moveit_fk(header, ['gripper_1_link'], rs).pose_stamped[0].pose
    grip2 = moveit_fk(header, ['gripper_2_link'], rs).pose_stamped[0].pose
    p1 = np.array([grip1.position.x, grip1.position.y, grip1.position.z])
    p2 = np.array([grip2.position.x, grip2.position.y, grip2.position.z])
    position = (p1+p2)/2
    return position

if __name__ == '__main__':
    rospy.init_node('rospy_log_loader',
                    anonymous=True)
    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', ]
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander("widowx_arm")
    j = group.get_active_joints()


    # header = Header(0, rospy.Time.now(), "/world")
    rospy.wait_for_service('compute_fk')
    try:
      moveit_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
    except rospy.ServiceException as e:
      rospy.logerror("Service call failed: %s"%e)
    rs = RobotState()
    rs.joint_state.name = joint_names

    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                        moveit_msgs.msg.DisplayTrajectory,
                                        queue_size=10)
    all_real_angles, all_pred_angles = get_angles_from_log('/home/robert/Research/Self-Modeling/logs/2018-08-01-00:07:08_log.txt')
    # all_real_angles, all_pred_angles = get_angles_from_log('/home/robert/Research/Self-Modeling/logs/2018-08-05-12:11:16_log.txt')

    print('')

    logged_drifts = []
    fk_drifts = []
    internal_drifts = []
    for n in range(len(all_real_angles)):
        angles_list = all_real_angles[n]
        # Linear Interpolation along angles for smoother/slower traveling (less wear on motors)
        int_len = 10
        for i in range(len(angles_list)-1):
            diff = angles_list[i*int_len+1]-angles_list[i*int_len]
            for j in range(1,int_len):
                val = angles_list[i*int_len]+float(j)*diff/float(int_len)
                angles_list.insert(i*int_len+j, val)
        if len(angles_list) < 120:
            # Converting data into ROS path for execution
            # Plan Setup
            plan = RobotTrajectory()
            plan.joint_trajectory.joint_names = joint_names
            # point = JointTrajectoryPoint()
            # point.positions = (angles[0], angles[1], angles[2], angles[3], angles[4])
            # plan.joint_trajectory.points.append(point)

            # Populating Plan
            i = 0
            for angles in angles_list:
                point = JointTrajectoryPoint()
                point.positions = (angles[0], angles[1], angles[2], angles[3], 0.0)
                plan.joint_trajectory.points.append(point)
                i += 1

            # joint_goal = group.get_current_joint_values()
            # joint_goal[0] = plan.joint_trajectory.points[0]
            # joint_goal[1] = -pi / 4
            # joint_goal[2] = 0
            # joint_goal[3] =
            # joint_goal[4] = 0

            # The go command can be called with joint values, poses, or without any
            # parameters if you have already set the pose or joint target for the group
            # curr = group.get_current_joint_values()
            # start = np.array(plan.joint_trajectory.points[0].positions)
            # reset_plan = group.plan(plan.joint_trajectory.points[0].positions)
            # group.execute(reset_plan)
            # Calling ``stop()`` ensures that there is no residual movement
            # group.stop()

            # Publishing to ROS
            print("============ Waiting while plan is loaded...")
            rospy.sleep(5)

            print("============ Publishing Plan")
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            display_trajectory_publisher.publish(display_trajectory)

            group.execute(plan, wait=True)
            group.stop()

            p1 = group.get_current_pose(end_effector_link='gripper_1_link')
            p1 = np.array([p1.pose.position.x, p1.pose.position.y, p1.pose.position.z])
            p2 = group.get_current_pose(end_effector_link='gripper_2_link')
            p2 = np.array([p2.pose.position.x, p2.pose.position.y, p2.pose.position.z])
            current_pose = (p1+p2)/2

            # print('Begin State')
            rs.joint_state.position = np.array([all_pred_angles[n][-1][0],
                                       all_pred_angles[n][-1][1],
                                       all_pred_angles[n][-1][2],
                                       all_pred_angles[n][-1][3],
                                       0.0])
            pos = get_end_effector_pos(moveit_fk, rs)
            print('Episode: '+str(n))
            print('Current Position: '+str(current_pose))
            print('Logged Pred Position: '+str(all_pred_angles[n][-1][-3:]))
            print('FK Pred Position'+str(pos))
            print('Logged Real Position: '+str(all_real_angles[n][-1][-3:]))
            print('')
            print('Logged Drift: '+str(np.linalg.norm(all_pred_angles[n][-1][-3:]-all_real_angles[n][-1][-3:])))
            print('FK Drift: '+str(np.linalg.norm(all_real_angles[n][-1][-3:]-pos)))
            print('Internal Drift: '+str(np.linalg.norm(all_pred_angles[n][-1][-3:]-pos)))
            logged_drifts.append(np.linalg.norm(all_pred_angles[n][-1][-3:]-all_real_angles[n][-1][-3:]))
            fk_drifts.append(np.linalg.norm(all_real_angles[n][-1][-3:]-pos))
            internal_drifts.append(np.linalg.norm(all_pred_angles[n][-1][-3:]-pos))
            print('')
            # print('At Position: '+str(current_pose))

            # raw_input('Press Enter to execute next path')
            # print("============ Waiting while plan is executed...")
            # rospy.sleep(5)
    import statistics
    print('Median Logged: '+str(statistics.median(logged_drifts)))
    print('Median FK: '+str(statistics.median(fk_drifts)))
    print('Median Internal: '+str(statistics.median(internal_drifts)))

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
    # obj = np.array([[1, 0, 0, 0],
    #                  [0, 1, 0, 0],
    #                  [0, 0, 1, 0],
    #                  [0, 0, 0, 1]]
    #                 )
    #
    # # Base Transform
    # obj = translate([0.0,0.0,0.125], obj)
    #
    # # Bicep FK
    # obj = rotate([0, -angles[1], -angles[0]], obj)
    # obj = translate([0.04825, 0, 0.14203], obj)
    # # print(obj[3][:3])
    #
    # # Forearm FK
    # obj = rotate([0, -pi/2-angles[2], 0], obj)
    # obj = translate([0, 0, 0.14203], obj)
    # # For New Forearm:
    # # obj = rotate([0, -pi/2-angles[2], 0], obj)
    # # obj = translate([0, h, 0.14203-0.08+0.073], obj)
    # # obj = rotate([0, -pi/6, 0], obj)
    # # obj = translate([0, h, 0.076], obj)
    # # print(obj[3][:3])
    #
    # # Wrist FK
    # obj = rotate([0, -angles[3], 0], obj)
    # obj = translate([0, 0, 0.11450], obj)
    #
    # # End Effector Position
    # return obj[3][:3]
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
    obj = translate([0, 0, 0.07], obj)
    # print(obj[3][:3])

    # Wrist FK
    obj = rotate([0, -angles[3], 0], obj)
    obj = translate([0, 0, 0.11450], obj)
    return obj[3][:3]

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
    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander("widowx_arm")
    gripper_group = moveit_commander.MoveGroupCommander("widowx_gripper")
    def reset_arm():
        group.execute(group.plan((math.pi/2, 0, 0, 0, 0)), wait='True')
    def open_gripper():
        gripper_group.execute(gripper_group.plan((0.031, 0.0)), wait='True')
    def close_gripper():
        gripper_group.execute(gripper_group.plan((0.012,0.0)), wait='True')
    display_trajectory_publisher = rospy.Publisher(
                                        '/move_group/display_planned_path',
                                        moveit_msgs.msg.DisplayTrajectory,
                                        queue_size=10)


    # master_plan = RobotTrajectory()
    # plans = pickle.load(open('closed_plan_data.pkl', 'rb+'))
    # master_plan.joint_trajectory.joint_names = joint_names
    #
    # for plan in plans:
    #     for point in plan.joint_trajectory.points:
    #         master_plan.joint_trajectory.points.append(point)
    #
    # print("============ Publishing Plan")
    # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    # display_trajectory.trajectory_start = robot.get_current_state()
    # display_trajectory.trajectory.append(master_plan)
    # display_trajectory_publisher.publish(display_trajectory)

    # master_plan = RobotTrajectory()
    # states = pickle.load(open('closed_listened_states.pkl', 'rb+'))
    # master_plan.joint_trajectory.joint_names = joint_names
    #
    # for state in states:
    #     point = JointTrajectoryPoint()
    #     point.positions = (state[0], state[1], state[2], state[3], 0.0)
    #     master_plan.joint_trajectory.points.append(point)
    #
    # print("============ Publishing Plan")
    # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    # display_trajectory.trajectory_start = robot.get_current_state()
    # display_trajectory.trajectory.append(master_plan)
    # display_trajectory_publisher.publish(display_trajectory)


    reset_arm()
    # open_gripper()
    close_gripper()

    # # Open
    # goal = control_msgs.msg.GripperCommandGoal()
    # goal.command.position = 0.04
    # client.send_goal_and_wait(goal, rospy.Duration(10))

    print "============ Printing robot state"
    print robot.get_current_state().joint_state.position
    print "============"

    tasks = pickle.load(open('custom_pick_and_place_open.pkl', 'rb+'))

    raw_input('Press Enter to Begin')

    # Task Plan
    pred_plans = []
    for i in range(len(tasks)):
        task = tasks[i]
        for state in task:
            angles = (float(state[0]), float(state[1]), float(state[2]), float(state[3]), 0.0)
            reset_plan = group.plan(angles)
            pred_plans.append(copy.copy(reset_plan))
            group.execute(reset_plan, wait=True)
            print "============ Printing robot state"
            real = 180*np.array(robot.get_current_state().joint_state.position)/math.pi
            pred = 180*np.array(angles)/math.pi
            # print(pred[:4]-real[:4])
            print(calc_end_effector_pos(np.radians(pred[:4]))-calc_end_effector_pos(np.radians(real[:4])))
            # print(calc_end_effector_pos(np.radians(pred[:4])))
            # print(calc_end_effector_pos(np.radians(real[:4])))
            print "============"
        # if i%3 == 1:
        #     close_gripper()
        # elif i%3 == 2:
        #     open_gripper()
        # inp = raw_input('Press Enter to Continue, q to quit: ')
        # if inp == 'q':
        #     break

    reset_plan = group.plan((math.pi/2, 0, 0, 0, 0))
    group.execute(reset_plan, wait='True')
    pred_plans.append(copy.copy(reset_plan))
    # reset_arm()

    # open_gripper()

    # pickle.dump(pred_plans, open('closed_plan_data.pkl','wb+'))



    # # Rand Data
    # import statistics, time
    # diffs = []
    # drifts = []
    # times = []
    # data = pickle.load(open('widowx_train_10hz_100K.pkl', 'rb+'))
    # index = 0
    # for i in range(10):
    #     above_ground=False
    #     while not above_ground:
    #         above_ground=True
    #         episode = []
    #         while True:
    #             episode.append(data[index])
    #             if data[index][0][-1] < 0.045 or data[index][3][-1] < 0.045: above_ground=False
    #             if data[index][4] == True:
    #                 index += 1
    #                 break
    #             index += 1
    #     reset_plan = group.plan((episode[i][0][0],episode[0][0][1], episode[0][0][2], episode[0][0][3], 0.0))
    #     group.execute(reset_plan, wait=True)
    #     start = time.time()
    #     for j in range(len(episode)):
    #         angles = (episode[j][3][0], episode[j][3][1], episode[j][3][2], episode[j][3][3], 0.0)
    #         group.execute(group.plan(angles), wait=True)
    #         real = 180*np.array(robot.get_current_state().joint_state.position)/math.pi
    #         pred = 180*np.array(angles)/math.pi
    #         diff = calc_end_effector_pos(np.radians(pred[:4]))-calc_end_effector_pos(np.radians(real[:4]))
    #         drift = np.linalg.norm(diff)
    #         # print(diff)
    #         print(drift)
    #         diffs.append(diff)
    #         drifts.append(drift)
    #     times.append(time.time()-start)
    #     print('')
    #     print('Index: '+str(index))
    #     print('Median Episode Time: '+str(statistics.median(times)))
    #     print('Median Drift: '+str(statistics.median(drifts)))
    #     print('Stdev Drift: '+str(statistics.stdev(drifts)))
    #     raw_input('Press Enter to Continue')



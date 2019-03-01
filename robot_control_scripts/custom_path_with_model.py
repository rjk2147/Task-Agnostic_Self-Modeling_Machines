import pickle
import numpy as np
from envs.widowx_arm import WidowxROS
from env_learners.dnn_env_learner import DNNEnvLearner
from test_planners.test_plan_widowx import find_next_move_test, find_next_move_train
import tensorflow as tf
import math

if __name__ == '__main__':
    loop = 'open'
    reset_target = np.array([0.171, 0.163, 0.225])
    targets = [

        # # 'H'
        # np.array([-0.02, 0.26, 0.07]),
        # np.array([-0.02, 0.20, 0.07]),
        # np.array([-0.02, 0.20, 0.10]),
        #
        # np.array([-0.02, 0.23, 0.07]),
        # np.array([0.0, 0.23, 0.07]),
        # np.array([0.0, 0.23, 0.10]),
        #
        # np.array([0.0, 0.26, 0.07]),
        # np.array([0.0, 0.20, 0.07]),
        # np.array([0.0, 0.20, 0.10]),
        #
        # # I
        # np.array([0.02, 0.26, 0.07]),
        # np.array([0.02, 0.20, 0.07]),
        # np.array([0.02, 0.20, 0.10]),
        # np.array([0.0, 0.30478, 0.26703]),

        # C Square
        np.array([-0.01, 0.26, 0.10]),
        np.array([-0.01, 0.24, 0.07]),
        np.array([-0.04, 0.24, 0.07]),
        # np.array([-0.02, 0.26, 0.10]),

        np.array([-0.04, 0.24, 0.07]),
        np.array([-0.04, 0.20, 0.07]),
        # np.array([-0.02, 0.20, 0.10]),

        np.array([-0.04, 0.20, 0.07]),
        np.array([-0.01, 0.20, 0.07]),
        np.array([-0.01, 0.20, 0.10]),

        # U Square
        np.array([0.01, 0.24, 0.07]),
        np.array([0.01, 0.20, 0.07]),
        # np.array([0.02, 0.20, 0.10]),

        np.array([0.01, 0.20, 0.065]),
        np.array([0.04, 0.20, 0.065]),
        # np.array([0.04, 0.20, 0.10]),

        np.array([0.04, 0.20, 0.07]),
        np.array([0.04, 0.24, 0.07]),
        # np.array([0.04, 0.20, 0.10]),


        # np.array([0.075, 0.25, 0.06]),
        # reset_target,
        # np.array([0.025, 0.25, 0.15]),
        # np.array([0.025, 0.25, 0.05]),
        # reset_target,
        # np.array([-0.025, 0.25, 0.15]),
        # np.array([-0.025, 0.25, 0.05]),
        # reset_target,
        #
        # np.array([0.075, 0.20, 0.15]),
        # np.array([0.075, 0.20, 0.05]),
        # reset_target,
        # np.array([0.025, 0.20, 0.15]),
        # np.array([0.025, 0.20, 0.05]),
        # reset_target,
        # np.array([-0.025, 0.20, 0.15]),
        # np.array([-0.025, 0.20, 0.05]),
        # reset_target,
        #
        # np.array([0.075, 0.15, 0.15]),
        # np.array([0.075, 0.15, 0.05]),
        # reset_target,
        # np.array([0.025, 0.15, 0.15]),
        # np.array([0.025, 0.15, 0.05]),
        # reset_target,
        # np.array([-0.025, 0.15, 0.15]),
        # np.array([-0.025, 0.15, 0.05]),
        # reset_target,
    ]
    # reset_target = np.array([0.171, 0.163, 0.18])
    # load = 'models/2018-08-05-12:11:16.ckpt' # 100K DNN Seq (5)
    # load = 'models/2018-08-15-19:24:23.ckpt' # 10K DNN Seq (5) Deformed Adaptation
    load = 'models/2018-08-18-07:31:09.ckpt' # 100K DNN Seq (5) Real
    env = WidowxROS()
    max_action = env.action_space.high
    env_learner = DNNEnvLearner(env)

    obs = env.reset(position=np.array([math.pi/2,0.0,0.0,0.0]))

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    try:
        saver = tf.train.Saver()
    except:
        saver=None

    paths = []
    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, load)
        print('Model: ' + load + ' Restored')
        env_learner.initialize(sess, load=True)

        print('')

        logged_drifts = []
        fk_drifts = []
        internal_drifts = []

        all_observations = []
        drifts = []
        for target in targets:
            path = []
            obs = env.reset(target=target, position=env.angles)
            real_d = np.linalg.norm(env.target - obs[-3:])
            done = False
            episode_step = 0
            while not done:
                # print(episode_step)
                if loop == 'real':
                    action = find_next_move_train(env, env_learner, obs, max_action, episode_step, dof=4)
                    real_obs, r, real_done, _ = env.step(max_action * action)
                    new_obs = real_obs
                else:
                    action = find_next_move_test(env, env_learner, obs, max_action, episode_step, dof=4)
                    new_obs = env_learner.step(obs, max_action * action, episode_step, save=True)
                    real_obs, r, real_done, _ = env.step(max_action * action)
                path.append(real_obs)
                d = np.linalg.norm(env.target - new_obs[-3:])
                # print(d)
                # print(action)
                # print(new_obs[-3:])
                # print('')
                real_d = np.linalg.norm(env.target - real_obs[-3:])
                if d < 0.01:
                    done = True
                if loop == 'open':
                    obs = new_obs
                elif loop == 'closed':
                    obs = real_obs
                elif loop == 'real':
                    obs = real_obs
                episode_step += 1
            print(real_obs[-3:])
            print('Dist From Target: '+str(real_d))
            print('')
            paths.append(path)
    print(len(paths))
    assert len(paths) == len(targets)
    pickle.dump(paths, open('custom_pick_and_place_'+loop+'.pkl','wb+'))

    # angles = [np.array([math.pi/2,0.0,0.0,0.0,0.0])]
    # for state in paths:
    #     angles.append(np.array([state[0], state[1], state[2], state[3], 0.0]))
    # angles.append(np.array([math.pi/2,0.0,0.0,0.0,0.0]))
    #
    # interp_angles = [np.array([math.pi/2,0.0,0.0,0.0,0.0])]
    # for i in range(20): interp_angles.append(interp_angles[0])
    # for i in range(1, len(angles)):
    #     diff = angles[i]-angles[i-1]
    #     m = np.max(np.abs(diff))
    #     nb_inp = int(round(m/0.02))
    #
    #     for j in range(1, nb_inp):
    #         interp_angle = angles[i-1]+float(j)*diff/float(nb_inp)
    #         interp_angles.append(interp_angle)
    #     interp_angles.append(angles[i])
    # # for i in range(20): interp_angles.append(interp_angles[-1])
    # #
    # import rospy
    # import moveit_msgs.msg
    # from moveit_msgs.msg._RobotTrajectory import RobotTrajectory
    # from trajectory_msgs.msg._JointTrajectoryPoint import JointTrajectoryPoint
    # from moveit_msgs.msg._RobotState import RobotState
    # display_trajectory_publisher = rospy.Publisher(
    #                                     '/move_group/display_planned_path',
    #                                     moveit_msgs.msg.DisplayTrajectory,
    #                                     queue_size=10)
    # plan = RobotTrajectory()
    # joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5']
    # rs = RobotState()
    # rs.joint_state.name = joint_names
    # rs.joint_state.position = interp_angles[0]
    # plan.joint_trajectory.joint_names = joint_names
    #
    # for angles in interp_angles:
    #     point = JointTrajectoryPoint()
    #     point.positions = angles
    #     plan.joint_trajectory.points.append(point)
    #
    # print("============ Publishing Plan")
    # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    # display_trajectory.trajectory_start = rs
    # display_trajectory.trajectory.append(plan)
    # display_trajectory_publisher.publish(display_trajectory)

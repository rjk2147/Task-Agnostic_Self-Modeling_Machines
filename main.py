import argparse
import logging
import time

import envs
import gym
import tensorflow as tf

import envs


def run(**kwargs):
    # if kwargs['env'] == 'simple_arm_3d':
    #     print('Environment \'simple_arm_3d\' chosen')
    #     from envs.simple_arm_3d import SimpleArm
    #     env = SimpleArm(train=True)
    #     from test_planners import test_plan_3d as testing
    # elif kwargs['env'] == 'improved_arm':
    #     print('Environment \'improved_arm\' chosen')
    #     from envs.improved_arm import ImprovedArm
    #     env = ImprovedArm(train=True)
    #     from test_planners import test_plan_3d as testing
    # elif kwargs['env'] == 'simple_arm_2d':
    #     print('Environment \'simple_arm_2d\' chosen')
    #     from envs.simple_arm import SimpleArm
    #     env = SimpleArm()
    #     from test_planners import test_plan as testing
    # elif kwargs['env'] == 'crawler':
    #     print('Environment \'crawler\' chosen')
    #     from envs.crawler import SimpleCrawler
    #     env = SimpleCrawler()
    #     from test_planners import test_plan_crawler as testing
    # elif kwargs['env'] == 'AntBulletEnv-v0':
    #     print('Environment \'AntBulletEnv-v0\' chosen')
    #     from test_planners import test_plan_walker as testing
    #     env = testing.AntWrapper(gym.make("AntBulletEnv-v0"))
    # elif kwargs['env'] == 'spring-mass-v0':
    #     print('Environment \'spring-mass-v0\' chosen')
    #     from envs.spring_mass_env import SpringMassCrawler
    #     env = SpringMassCrawler()
    #     from test_planners import test_spring_mass as testing
    if kwargs['env'] == 'widowx_arm':
        print('Environment \'widowx_arm\' chosen')
        from envs.widowx_arm import WidowxROS
        env = WidowxROS()
        from test_planners import test_plan_widowx as testing
    else:
        print('No valid environment chosen')
        print('Defaulting to simple_arm_3d')
        from envs.widowx_arm import WidowxROS
        env = WidowxROS()
        from test_planners import test_plan_widowx as testing

    tf.reset_default_graph()

    if kwargs['arch'] == 'knn':
        print('KNN architecture chosen')
        from env_learners.knn_env_learner import KNNEnvLearner
        env_learner = KNNEnvLearner(env)
    elif kwargs['arch'] == 'dnn':
        print('Traditional Deep Neural Network architecture chosen')
        from env_learners.dnn_env_learner import DNNEnvLearner
        env_learner = DNNEnvLearner(env)
    else:
        print('No valid architecture chosen')
        print('Defaulting to \'dnn\'')
        from env_learners.dnn_env_learner import DNNEnvLearner
        env_learner = DNNEnvLearner(env)

    start_time = time.time()
    testing.test(env=env, env_learner=env_learner, epochs=kwargs['nb_epochs'], train_episodes=kwargs['nb_train_episodes'], load=kwargs['load'],
                 test_episodes=kwargs['nb_test_episodes'], loop=kwargs['loop'])
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='widowx_arm')
    parser.add_argument('--arch', type=str, default='dnn')
    parser.add_argument('--loop', type=str, default='open')
    parser.add_argument('--nb-epochs', type=int, default=100)
    parser.add_argument('--nb-train-episodes', type=int, default=100)
    parser.add_argument('--nb-test-episodes', type=int, default=100)
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    # Run actual script.
    run(**args)

import argparse
import logging
import time
import gym
import tensorflow as tf

import pybullet as p
import pybullet_envs
import simple_arm
import simple_arm_3d
import improved_arm
import crawler
import spring_mass_env
import logger

def run(**kwargs):
    if kwargs['env'] == 'simple_arm_3d':
        print('Environment \'simple_arm_3d\' chosen')
        env = simple_arm_3d.SimpleArm(train=True)
        import test_plan_3d as testing
    if kwargs['env'] == 'improved_arm':
        print('Environment \'improved_arm\' chosen')
        env = improved_arm.ImprovedArm(train=True)
        import test_plan_3d as testing
    elif kwargs['env'] == 'simple_arm_2d':
        print('Environment \'simple_arm_2d\' chosen')
        env = simple_arm.SimpleArm()
        import test_plan as testing
    elif kwargs['env'] == 'crawler':
        print('Environment \'crawler\' chosen')
        env = crawler.SimpleCrawler()
        import test_plan_crawler as testing
    elif kwargs['env'] == 'AntBulletEnv-v0':
        print('Environment \'AntBulletEnv-v0\' chosen')
        import test_plan_walker as testing
        env = testing.AntWrapper(gym.make("AntBulletEnv-v0"))
    elif kwargs['env'] == 'spring-mass-v0':
        print('Environment \'spring-mass-v0\' chosen')
        env = spring_mass_env.SpringMassCrawler()
        import test_spring_mass as testing
    else:
        print('No valid environment chosen')
        print('Defaulting to simple_arm_3d')
        env = simple_arm_3d.SimpleArm(train=True)
        import test_plan_3d as testing
    gym.logger.setLevel(logging.WARN)

    logger.info('logdir={}'.format(logger.get_dir()))
    tf.reset_default_graph()

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()
    testing.test(env=env, epochs=kwargs['nb_epochs'], train_episodes=kwargs['nb_train_episodes'], load=kwargs['load'],
                 test_episodes=kwargs['nb_test_episodes'], loop=kwargs['loop'], show_model=kwargs['show_model'])
    env.close()
    logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='improved_arm')
    parser.add_argument('--loop', type=str, default='open')
    parser.add_argument('--nb-epochs', type=int, default=100)
    parser.add_argument('--nb-train-episodes', type=int, default=100)
    parser.add_argument('--nb-test-episodes', type=int, default=100)
    parser.add_argument('--show-model', dest='show_model', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    parser.set_defaults(show_model=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    logger.configure()
    # Run actual script.
    run(**args)

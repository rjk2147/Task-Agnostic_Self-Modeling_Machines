import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from std_msgs.msg import String

def move_group_python_interface_tutorial():
  ## BEGIN_TUTORIAL
  ##
  ## Setup
  ## ^^^^^
  ## CALL_SUB_TUTORIAL imports
  ##
  ## First initialize moveit_commander and rospy.
  print "============ Starting tutorial setup"
  moveit_commander.roscpp_initialize(sys.argv)
  rospy.init_node('move_group_python_interface_tutorial',
                  anonymous=True)

  ## Instantiate a RobotCommander object.  This object is an interface to
  ## the robot as a whole.
  robot = moveit_commander.RobotCommander()

  ## Instantiate a PlanningSceneInterface object.  This object is an interface
  ## to the world surrounding the robot.
  scene = moveit_commander.PlanningSceneInterface()

  ## Instantiate a MoveGroupCommander object.  This object is an interface
  ## to one group of joints.  In this case the group is the joints in the left
  ## arm.  This interface can be used to plan and execute motions on the left
  ## arm.
  group = moveit_commander.MoveGroupCommander("widowx_arm")


  ## We create this DisplayTrajectory publisher which is used below to publish
  ## trajectories for RVIZ to visualize.
  display_trajectory_publisher = rospy.Publisher(
                                      '/move_group/display_planned_path',
                                      moveit_msgs.msg.DisplayTrajectory)

  ## Wait for RVIZ to initialize. This sleep is ONLY to allow Rviz to come up.
  print "============ Waiting for RVIZ..."
  rospy.sleep(10)
  print "============ Starting tutorial "

  ## Getting Basic Information
  ## ^^^^^^^^^^^^^^^^^^^^^^^^^
  ##
  ## We can get the name of the reference frame for this robot
  print "============ Reference frame: %s" % group.get_planning_frame()

  ## We can also print the name of the end-effector link for this group
  print "============ Reference frame: %s" % group.get_end_effector_link()

  ## We can get a list of all the groups in the robot
  print "============ Robot Groups:"
  print robot.get_group_names()

  ## Sometimes for debugging it is useful to print the entire state of the
  ## robot.
  print "============ Printing robot state"
  print robot.get_current_state()
  print "============"


  ## Planning to a Pose goal
  ## ^^^^^^^^^^^^^^^^^^^^^^^
  ## We can plan a motion for this group to a desired pose for the 
  ## end-effector
  print "============ Generating plan 1"
  pose_target = geometry_msgs.msg.Pose()
  pose_target.orientation.w = 1.0
  pose_target.position.x = 0.0
  pose_target.position.y = 0.25
  pose_target.position.z = 0.1
  group.set_pose_target(pose_target)

  ## Now, we call the planner to compute the plan
  ## and visualize it if successful
  ## Note that we are just planning, not asking move_group 
  ## to actually move the robot
  plan1 = group.plan()

  print "============ Waiting while RVIZ displays plan1..."
  rospy.sleep(5)

 
  ## You can ask RVIZ to visualize a plan (aka trajectory) for you.  But the
  ## group.plan() method does this automatically so this is not that useful
  ## here (it just displays the same trajectory again).
  print "============ Visualizing plan1"
  display_trajectory = moveit_msgs.msg.DisplayTrajectory()

  display_trajectory.trajectory_start = robot.get_current_state()
  display_trajectory.trajectory.append(plan1)
  display_trajectory_publisher.publish(display_trajectory);

  print "============ Waiting while plan1 is visualized (again)..."
  rospy.sleep(5)


  ## Moving to a pose goal
  ## ^^^^^^^^^^^^^^^^^^^^^
  ##
  ## Moving to a pose goal is similar to the step above
  ## except we now use the go() function. Note that
  ## the pose goal we had set earlier is still active 
  ## and so the robot will try to move to that goal. We will
  ## not use that function in this tutorial since it is 
  ## a blocking function and requires a controller to be active
  ## and report success on execution of a trajectory.

  # Uncomment below line when working with a real robot
  # group.go(wait=True)

  ## Planning to a joint-space goal 
  ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ##
  ## Let's set a joint space goal and move towards it. 
## First, we will clear the pose target we had just set.
move_group_python_interface_tutorial()

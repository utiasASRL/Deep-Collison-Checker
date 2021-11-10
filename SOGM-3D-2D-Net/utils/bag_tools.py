import rospy
import numpy
from ros_numpy import point_cloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
import yaml
import tf2_geometry_msgs
import tf.transformations


def bag_metadata(bagfile):
    res = yaml.load(bagfile._get_yaml_info())
    return res


def trajectory_to_array(traj, time_origin=0):
    ''' given a list of PoseStamped messages, return a structured numpy array'''
    arr = []
    for msg in traj:
        pose = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x,
                msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w, msg.header.stamp.to_sec() - time_origin)
        arr.append(pose)

    arr = numpy.array(arr, dtype=[('pos_x', numpy.double), ('pos_y', numpy.double), ('pos_z', numpy.double), ('rot_x', numpy.double),
                      ('rot_y', numpy.double), ('rot_z', numpy.double), ('rot_w', numpy.double), ('time', numpy.double)])
    return arr


def num_messages(topic_name, bagfile):
    count = 0
    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):
        count += 1

    return count


def read_pointcloud_frames(topic_name, bagfile):
    '''returns a list of tuples: (stamp,numpy_array) where stamp is the time and numpy_array is a labelled array with the pointcloud data'''
    frames = []

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):
        pc_array = point_cloud2.pointcloud2_to_array(msg)
        frames.append((msg.header.stamp, pc_array))

    return frames


def read_collider_preds(topic_name, bagfile):
    '''returns list of Voxgrid message'''

    #from teb_local_planner.msg import VoxGrid

    all_header_stamp = []
    all_dims = []
    all_origin = []
    all_dl = []
    all_dt = []
    all_preds = []

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):

        all_header_stamp.append(msg.header.stamp.to_sec())
        all_dims.append([msg.depth, msg.width, msg.height])
        all_origin.append([msg.origin.x, msg.origin.y, msg.origin.z])
        all_dl.append(msg.dl)
        all_dt.append(msg.dt)
        array_data = numpy.frombuffer(msg.data, dtype=numpy.uint8)
        all_preds.append(array_data.tolist())

    collider_data = {}

    collider_data['header_stamp'] = numpy.array(all_header_stamp, dtype=numpy.float64)
    collider_data['dims'] = numpy.array(all_dims, dtype=numpy.int32)
    collider_data['origin'] = numpy.array(all_origin, dtype=numpy.float32)
    collider_data['dl'] = numpy.array(all_dl, dtype=numpy.float32)
    collider_data['dt'] = numpy.array(all_dt, dtype=numpy.float32)
    collider_data['preds'] = numpy.array(all_preds, dtype=numpy.uint8)

    return collider_data


def read_local_plans(topic_name, bagfile):
    '''returns list of local plan message'''

    local_plans = []

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):

        path_dict = {}
        path_dict['header_stamp'] = msg.header.stamp.to_sec()
        path_dict['header_frame_id'] = msg.header.frame_id

        pose_list = []
        for msg_pose in msg.poses:
            pose = (msg_pose.pose.position.x,
                    msg_pose.pose.position.y,
                    msg_pose.pose.position.z,
                    msg_pose.pose.orientation.x,
                    msg_pose.pose.orientation.y,
                    msg_pose.pose.orientation.z,
                    msg_pose.pose.orientation.w)
            pose_list.append(pose)

        path_dict['pose_list'] = pose_list

        local_plans.append(path_dict)

    return local_plans


def read_nav_odometry(topic_name, bagfile, nav_msg=True):
    '''returns an array of geometry_msgs/PoseStamped'''

    arr = []

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):

        if (not nav_msg):
            arr.append(msg)
        else:
            geo_msg = PoseStamped()
            geo_msg.header = msg.header
            geo_msg.pose = msg.pose.pose
            arr.append(geo_msg)
            #pose = (msg.pose.pose.position.x,
            # msg.pose.pose.position.y,
            # msg.pose.pose.position.z,
            # msg.pose.pose.orientation.x,
            # msg.pose.pose.orientation.y,
            # msg.pose.pose.orientation.z,
            # msg.pose.pose.orientation.w,
            # msg.header.stamp.to_sec())
            #arr.append(pose)

    #arr = numpy.array(arr, dtype = [('pos_x',numpy.double),
    # ('pos_y',numpy.double),
    # ('pos_z',numpy.double),
    # ('rot_x',numpy.double),
    # ('rot_y',numpy.double),
    # ('rot_z',numpy.double),
    # ('rot_w',numpy.double),
    # ('time',numpy.double)])
    return arr


def read_tf_transform(parent_frame, child_frame, bagfile, static=False):
    ''' returns a list of time stamped transforms between parent frame and child frame '''
    arr = []
    if (static):
        topic_name = "/tf_static"
    else:
        topic_name = "/tf"

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):
        for transform in msg.transforms:
            if (transform.header.frame_id == parent_frame and transform.child_frame_id == child_frame):
                arr.append(transform)

    return arr


def read_action_result(topic_name, bagfile):
    ''' reads move_base action result info. Returns an array of the times and status of each message '''

    arr = []

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):
        arr.append((msg.header.stamp.to_sec(), msg.status.status))

    arr = numpy.array(arr, dtype=[('time', numpy.double), ('status', numpy.int8)])

    return arr


def transforms_to_trajectory(transforms):
    ''' converts a list of time stamped transforms to a list of pose stamped messages, where the pose is that of the child frame relative to it's parent'''
    traj = []
    for transform in transforms:
        geo_msg = PoseStamped()
        geo_msg.header = transform.header
        geo_msg.header.frame_id = transform.child_frame_id
        geo_msg.pose.position = transform.transform.translation
        geo_msg.pose.orientation = transform.transform.rotation
        traj.append(geo_msg)

    return traj


def date_to_int(date):
    ''' returns a tuple (int represenaiton of date, [years, months, day ...) '''
    d_list = date.split('-')
    split_date = {}
    labels = ['year', 'month', 'day,', 'hour', 'minute', 'second']
    conversions = [31556952, 2629746, 86400, 3600, 60, 1]
    res = 0
    for i in range(len(d_list)):
        split_date[labels[i]] = int(d_list[i])
        res += conversions[i] * int(d_list[i])

    return (res, split_date)


def get_interpolations(target_times, trajectory, transform=True):
    '''
    given two trajectories, interploate the poses in trajectory to the times given in target_times (another trajectory)
    this modifies target_times so it stays in the range of trajectory's interpolations
    if transform = True, then trajectory stores transform messages not PoseStamped
    '''

    min_time = trajectory[0].header.stamp.to_sec()
    max_time = trajectory[-1].header.stamp.to_sec()

    res = []

    last = 0

    i = 0
    while i < len(target_times):

        target = target_times[i]
        time = target.header.stamp.to_sec()

        if (time < min_time or time > max_time):
            target_times.pop(i)
            continue

        lower_ind = last

        while (trajectory[lower_ind].header.stamp.to_sec() > time or trajectory[lower_ind + 1].header.stamp.to_sec() < time):
            if (trajectory[lower_ind].header.stamp.to_sec() > time):
                lower_ind -= 1
            else:
                lower_ind += 1

        #last = lower_ind +1

        if ((i + 1) < len(target_times)):
            next_time = target_times[i + 1].header.stamp.to_sec()
            if (next_time >= trajectory[lower_ind + 1].header.stamp.to_sec()):
                last = lower_ind + 1
            else:
                last = lower_ind
        else:
            last = lower_ind

        #last = (lower_ind+1) if ((lower_ind+2)<len(trajectory)) else lower_ind

        if (transform):
            inter = interpolate_transform(time, trajectory[lower_ind], trajectory[lower_ind + 1])
        else:
            inter = interpolate_pose(time, trajectory[lower_ind], trajectory[lower_ind + 1])

        res.append(inter)
        i += 1

    return res


def transform_trajectory(trajectory, transformations):
    ''' translate each point in trajectory by a transformation interpolated to the correct time, return the transformed trajectory'''

    # for each point in trajectory, find the interpolated transformation, then transform the trajectory point

    matching_transforms = get_interpolations(trajectory, transformations)

    res = []

    for i in range(len(matching_transforms)):
        trans = matching_transforms[i]
        traj_pose = trajectory[i]

        transformed = tf2_geometry_msgs.do_transform_pose(traj_pose, trans)
        res.append(transformed)

    return res


def interpolate_pose(time, pose1, pose2):
    ''' given a target time, and two PoseStamped messages, find the interpolated pose between pose1 and pose2 '''

    t1 = pose1.header.stamp.to_sec()
    t2 = pose2.header.stamp.to_sec()

    alpha = 0
    if (t1 != t2):
        alpha = (time - t1) / (t2 - t1)

    pos1 = pose1.pose.position
    pos2 = pose2.pose.position

    rot1 = pose1.pose.orientation
    rot1 = [rot1.x, rot1.y, rot1.z, rot1.w]
    rot2 = pose2.pose.orientation
    rot2 = [rot2.x, rot2.y, rot2.z, rot2.w]

    res = PoseStamped()

    res.header.stamp = rospy.Time(time)
    res.header.frame_id = pose1.header.frame_id

    res.pose.position.x = pos1.x + (pos2.x - pos1.x) * alpha
    res.pose.position.y = pos1.y + (pos2.y - pos1.y) * alpha
    res.pose.position.z = pos1.z + (pos2.z - pos1.z) * alpha

    res_rot = tf.transformations.quaternion_slerp(rot1, rot2, alpha)
    res.pose.orientation.x = res_rot[0]
    res.pose.orientation.y = res_rot[1]
    res.pose.orientation.z = res_rot[2]
    res.pose.orientation.w = res_rot[3]

    return res


def interpolate_transform(time, trans1, trans2):
    ''' given a target time, and two TransformStamped messages, find the interpolated transform '''

    t1 = trans1.header.stamp.to_sec()
    t2 = trans2.header.stamp.to_sec()

    alpha = 0
    if (t1 != t2):
        alpha = (time - t1) / (t2 - t1)

    pos1 = trans1.transform.translation
    pos2 = trans2.transform.translation

    rot1 = trans1.transform.rotation
    rot1 = [rot1.x, rot1.y, rot1.z, rot1.w]
    rot2 = trans2.transform.rotation
    rot2 = [rot2.x, rot2.y, rot2.z, rot2.w]

    res = TransformStamped()

    res.header.stamp = rospy.Time(time)
    res.header.frame_id = trans1.header.frame_id

    res.transform.translation.x = pos1.x + (pos2.x - pos1.x) * alpha
    res.transform.translation.y = pos1.y + (pos2.y - pos1.y) * alpha
    res.transform.translation.z = pos1.z + (pos2.z - pos1.z) * alpha

    res_rot = tf.transformations.quaternion_slerp(rot1, rot2, alpha)
    res.transform.rotation.x = res_rot[0]
    res.transform.rotation.y = res_rot[1]
    res.transform.rotation.z = res_rot[2]
    res.transform.rotation.w = res_rot[3]

    return res


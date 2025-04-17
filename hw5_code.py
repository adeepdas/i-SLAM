import numpy as np
import gtsam
from typing import NamedTuple

def pretty_print(arr):
    return '\n'.join([' '.join(['%.2f' % x for x in c]) for c in arr])

class Pose2(NamedTuple):
    '''
    Pose2 class for 2D pose
    @usage: pose = Pose2(id, x, y, z)
            print(pose.x)
    '''
    id: int
    x: float
    y: float
    theta: float

class Edge2(NamedTuple):
    '''
    Edge2 class for 2D edge
    @usage: edge = Edge2(id1, id2, x, y, z, info)
            print(edge.x)
    '''
    id1: int
    id2: int
    x: float
    y: float
    theta: float
    info: np.ndarray # 3x3 matrix

    def __str__(self):
        return f"Edge2(id1={self.id1}, id2={self.id2}, x={self.x}, y={self.y}, theta={self.theta},\ninfo=\n{pretty_print(self.info)})\n"

class Pose3(NamedTuple):
    '''
    Pose3 class for 3D pose
    @usage: pose = Pose3(id, x, y, z, qx, qy, qz, qw)
            print(pose.x)
    '''
    id: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

class Edge3(NamedTuple):
    '''
    Edge3 class for 3D edge
    @usage: edge = Edge3(id1, id2, x, y, z, qx, qy, qz, qw, info)
            print(edge.x)
    '''
    id1: int
    id2: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    info: np.ndarray # 6x6 matrix

    def __str__(self):
        return f"Edge3(id1={self.id1}, id2={self.id2}, x={self.x}, y={self.y}, z={self.z}, qx={self.qx}, qy={self.qy}, qz={self.qz}, qw={self.qw},\ninfo=\n{pretty_print(self.info)})\n"


def read_g2o_2d(file_name):
    data = {
        'poses': [],
        'edges': []
    }

    # read the file
    with open(file_name, "r") as f:
        lines = f.readlines()

        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################

        # fill in the `data` dict with Pose2 or Edge2 objects
        
        for line in lines:
            line = line.strip()
            if line.startswith("VERTEX_SE2"):
                parts = line.split()
                i = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                theta = float(parts[4])
                data['poses'].append(Pose2(i, x, y, theta))
            elif line.startswith("EDGE_SE2"):
                parts = line.split()
                i = int(parts[1])
                j = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                theta = float(parts[5])
                info_values = [float(val) for val in parts[6:]]
                info = np.array([
                    [info_values[0], info_values[1], info_values[2]],
                    [info_values[1], info_values[3], info_values[4]],
                    [info_values[2], info_values[4], info_values[5]]
                ])
                data['edges'].append(Edge2(i, j, x, y, theta, info))         
    
        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
    return data

def gn_2d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()
    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create an empty factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    # set initial_values according to poses
    for pose in poses:
        initial_values.insert(pose.id, gtsam.Pose2(pose.x, pose.y, pose.theta))

    # add prior factor for the first pose
    first_pose = poses[0]
    prior_factor = gtsam.PriorFactorPose2(
        first_pose.id,
        gtsam.Pose2(first_pose.x, first_pose.y, first_pose.theta),
        gtsam.noiseModel.Gaussian.Covariance(np.diag(first_pose_prior_cov)))
    graph.add(prior_factor)

    # add between factors according to edges
    for edge in edges:
        noise_model = gtsam.noiseModel.Gaussian.Covariance(np.linalg.inv(edge.info))
        between_factor = gtsam.BetweenFactorPose2(
            edge.id1,
            edge.id2,
            gtsam.Pose2(edge.x, edge.y, edge.theta),
            noise_model)
        graph.add(between_factor)

    # optimize the graph
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_values)
    result = optimizer.optimize()
    
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose2(result)

def isam_2d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()
    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create optimizer
    isam = gtsam.ISAM2()     

    for pose in poses:

        frame_id = pose.id

        # create an empty factor graph
        graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()
        
        if frame_id==0:
            # initialization
            prior_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag(first_pose_prior_cov))
            graph.add(gtsam.PriorFactorPose2(
                frame_id,
                gtsam.Pose2(pose.x, pose.y, pose.theta),
                prior_noise))
            initial_values.insert(frame_id, gtsam.Pose2(pose.x, pose.y, pose.theta))
        else:
            # optimize new frame
            prev_pose = result.atPose2(frame_id - 1)
            initial_values.insert(frame_id, prev_pose)

            for edge in edges:
                if edge.id2 == frame_id:
                    cov = np.linalg.inv(edge.info)  
                    model = gtsam.noiseModel.Gaussian.Covariance(cov)
                    graph.add(gtsam.BetweenFactorPose2(
                        edge.id1,
                        edge.id2,
                        gtsam.Pose2(edge.x, edge.y, edge.theta),
                        model))

        # update isam
        isam.update(graph, initial_values)
        result = isam.calculateEstimate()

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose2(result)

def read_g2o_3d(file_name):
    data = {
        'poses': [],
        'edges': []
    }


    # read the file
    with open(file_name, "r") as f:
        lines = f.readlines()

        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################

        # fill in the `data` dict with Pose3 or Edge3 objects

        for line in lines:
            line = line.strip()
            if line.startswith("VERTEX_SE3:QUAT"):
                parts = line.split()
                i = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                qx = float(parts[5])
                qy = float(parts[6])
                qz = float(parts[7])
                qw = float(parts[8])
                data['poses'].append(Pose3(i, x, y, z, qx, qy, qz, qw))
            elif line.startswith("EDGE_SE3:QUAT"):
                parts = line.split()
                i = int(parts[1])
                j = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
                qx = float(parts[6])
                qy = float(parts[7])
                qz = float(parts[8])
                qw = float(parts[9])
                info_values = [float(val) for val in parts[10:]]
                info = np.array([
                    [info_values[0],  info_values[1],  info_values[2],  info_values[3],  info_values[4],  info_values[5]],
                    [info_values[1],  info_values[6],  info_values[7],  info_values[8],  info_values[9],  info_values[10]],
                    [info_values[2],  info_values[7],  info_values[11], info_values[12], info_values[13], info_values[14]],
                    [info_values[3],  info_values[8],  info_values[12], info_values[15], info_values[16], info_values[17]],
                    [info_values[4],  info_values[9],  info_values[13], info_values[16], info_values[18], info_values[19]],
                    [info_values[5],  info_values[10], info_values[14], info_values[17], info_values[19], info_values[20]]
                ])
                data['edges'].append(Edge3(i, j, x, y, z, qx, qy, qz, qw, info))
            
        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
    
    return data

def gn_3d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()

    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create an empty factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    # set initial_values according to poses
    for pose in poses:
        initial_values.insert(
            pose.id,
            gtsam.Pose3(gtsam.Rot3.Quaternion(pose.qw, pose.qx, pose.qy, pose.qz), gtsam.Point3(pose.x, pose.y, pose.z)))

    # add prior factor for the first pose
    first_pose = poses[0]
    prior_factor = gtsam.PriorFactorPose3(
        first_pose.id,
        gtsam.Pose3(gtsam.Rot3.Quaternion(first_pose.qw, first_pose.qx, first_pose.qy, first_pose.qz),
                    gtsam.Point3(first_pose.x, first_pose.y, first_pose.z)),
        gtsam.noiseModel.Gaussian.Covariance(np.diag(first_pose_prior_cov)))
    graph.add(prior_factor)

    # add between factors according to edges
    for edge in edges:
        noise_model = gtsam.noiseModel.Gaussian.Covariance(np.linalg.inv(edge.info))
        between_factor = gtsam.BetweenFactorPose3(
            edge.id1,
            edge.id2,
            gtsam.Pose3(gtsam.Rot3.Quaternion(edge.qw, edge.qx, edge.qy, edge.qz), gtsam.Point3(edge.x, edge.y, edge.z)),
            noise_model)
        graph.add(between_factor)

    # optimize the graph
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_values)
    result = optimizer.optimize()

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose3(result)

def isam_3d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()

    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create optimizer
    isam = gtsam.ISAM2() 

    for pose in poses:

        frame_id = pose.id

        # create an empty factor graph
        graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()
        
        if frame_id==0:
            # initialization

            # Notice that the order of quaternion is different from
            # the one in the g2o file. GTSAM uses (qw, qx, qy, qz).
            prior_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag(first_pose_prior_cov))
            graph.add(gtsam.PriorFactorPose3(
                frame_id,
                gtsam.Pose3(
                    gtsam.Rot3.Quaternion(pose.qw, pose.qx, pose.qy, pose.qz),
                    gtsam.Point3(pose.x, pose.y, pose.z)), prior_noise))
            initial_values.insert(
                frame_id,
                gtsam.Pose3(
                    gtsam.Rot3.Quaternion(pose.qw, pose.qx, pose.qy, pose.qz),
                    gtsam.Point3(pose.x, pose.y, pose.z)))
        else:
            # optimize new frame
            prev_pose = result.atPose3(frame_id - 1)
            initial_values.insert(frame_id, prev_pose)

            for edge in edges:
                if edge.id2 == frame_id:
                    cov = np.linalg.inv(edge.info)  
                    model = gtsam.noiseModel.Gaussian.Covariance(cov)
                    graph.add(gtsam.BetweenFactorPose3(
                        edge.id1,
                        edge.id2,
                        gtsam.Pose3(
                            gtsam.Rot3.Quaternion(edge.qw, edge.qx, edge.qy, edge.qz),
                            gtsam.Point3(edge.x, edge.y, edge.z)), model))

        # update isam
        isam.update(graph, initial_values)
        result = isam.calculateEstimate()

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose3(result)

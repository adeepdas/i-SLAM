import numpy as np
import gtsam
from gtsam import symbol
from visual_odometry import extract_visual_odometry
from imu_integration import read_imu_data
from visualization import animate_trajectory

def batch_optimization(imu_data, video_data):
    """
    Perform batch optimization of SLAM trajectory using GTSAM with visual odometry and IMU preintegration.
    
    Args:
        imu_data (dict): IMU data
        video_data (dict): RGBD video data

    Returns:
        refined_transforms (np.ndarray): Refined transformation matrices of shape (N, 4, 4)
    """
    print("Performing visual odometry...")
    video_timestamps = video_data['timestamp']
    vo_transforms = extract_visual_odometry(video_data)
    
    print("Reading IMU data...")
    imu_timestamps, acc_data, gyro_data = read_imu_data(imu_data)

    print("Creating factor graph...")
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    pose_key = lambda t: symbol('x', t)
    vel_key = lambda t: symbol('v', t)
    bias_key = lambda t: symbol('b', t)

    print("Configuring noise models...")
    # IMU noise model
    acc_noise_sigma = 0.1
    gyro_noise_sigma = 0.01
    imu_params = gtsam.PreintegrationParams.MakeSharedU(0.0)  # gravity set to 0 for now
    imu_params.setAccelerometerCovariance(np.eye(3) * acc_noise_sigma**2)
    imu_params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2)
    imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)
    initial_bias = gtsam.imuBias.ConstantBias(biasAcc=np.zeros(3), biasGyro=np.zeros(3))
    preint_imu = gtsam.PreintegratedImuMeasurements(imu_params, initial_bias)
    # pose prior noise model
    first_pose_prior_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    prior_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag(first_pose_prior_cov))
    # VO noise model (for optional VO factors)
    vo_cov = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    vo_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag(vo_cov))

    print("Initializing factor graph...")
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    identity_pose = gtsam.Pose3()
    initial_values.insert(pose_key(0), identity_pose)
    initial_values.insert(vel_key(0), np.zeros(3))
    initial_values.insert(bias_key(0), initial_bias)
    graph.add(gtsam.PriorFactorPose3(pose_key(0), identity_pose, prior_noise))
    graph.add(gtsam.PriorFactorVector(vel_key(0), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)))
    graph.add(gtsam.PriorFactorConstantBias(bias_key(0), initial_bias, gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)))

    print("Adding factors to graph...")
    prev_imu_idx = 0
    for t in range(1, len(vo_transforms)):
        # add VO factor
        R_vo = vo_transforms[t, :3, :3]
        t_vo = vo_transforms[t, :3, -1]
        pose_vo = gtsam.Pose3(gtsam.Rot3(R_vo), gtsam.Point3(t_vo))
        graph.add(gtsam.BetweenFactorPose3(pose_key(t-1), pose_key(t), pose_vo, vo_noise))

        # add IMU factor
        preint_imu.resetIntegration()
        i = prev_imu_idx
        num_measurements = 0
        while i < len(imu_timestamps) and imu_timestamps[i] <= video_timestamps[t]:
            dt = imu_timestamps[i] - imu_timestamps[i-1] if i > 0 else 0.01
            if not np.isinf(dt) and dt > 0 and dt < 0.5:
                preint_imu.integrateMeasurement(acc_data[i], gyro_data[i], dt)
                num_measurements += 1
            i += 1
        if num_measurements == 0:
            print(f"No IMU data for frame: {t}")
            # add indentity constraint if no IMU data between frames to prevent unconstrained optimization
            preint_imu.integrateMeasurement(np.zeros(3), np.zeros(3), 0.01)
        prev_imu_idx = i

        # add IMU factor
        graph.add(gtsam.ImuFactor(
            pose_key(t-1), vel_key(t-1),
            pose_key(t), vel_key(t),
            bias_key(t-1),
            preint_imu
        ))

        # add bias factor
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        graph.add(gtsam.BetweenFactorConstantBias(
            bias_key(t-1), bias_key(t),
            gtsam.imuBias.ConstantBias(),  # delta = 0 → constant bias assumption
            bias_noise
        ))

        # add initial values
        initial_values.insert(pose_key(t), identity_pose)
        initial_values.insert(vel_key(t), np.zeros(3))
        initial_values.insert(bias_key(t), initial_bias)
    
    print("Optimizing pose graph...")
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_values)
    result = optimizer.optimize()
    
    print("Extracting optimized poses...")
    refined_poses = gtsam.utilities.extractPose3(result)
    
    refined_transforms = []
    for pose in refined_poses:
        R = pose[:-3].reshape(3, 3)
        t = pose[-3:]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        refined_transforms.append(T)
    
    return np.array(refined_transforms)

if __name__ == "__main__":
    imu_data = np.load('data/v2/imu_data_straight_line.npy', allow_pickle=True)
    video_data = np.load('data/v2/video_data_straight_line.npy', allow_pickle=True)
    
    refined_transforms = batch_optimization(imu_data, video_data)
    
    print("Animating trajectory...")
    orientations = refined_transforms[:, :3, :3]
    positions = refined_transforms[:, :3, -1]
    animate_trajectory(orientations, positions)
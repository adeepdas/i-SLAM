import numpy as np
import gtsam
from gtsam import symbol
from iLoco.visual_odometry import extract_visual_odometry
from iLoco.imu_integration import read_imu_data
from iLoco.visualization import animate_trajectory, plot_trajectory

def graph_optimization(imu_data, video_data, mini_batch_size=10):
    """
    Perform batch optimization of SLAM trajectory using GTSAM with visual odometry and IMU preintegration.
    
    Args:
        imu_data (dict): IMU data
        video_data (dict): RGBD video data
        mini_batch_size (int): Number of frames to process in each batch

    Returns:
        refined_transforms (np.ndarray): Refined transformation matrices of shape (N, 4, 4)
    """
    print("Performing visual odometry...")
    vo_timestamps, vo_transforms = extract_visual_odometry(video_data, min_matches=50)
    
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
    isam = gtsam.ISAM2()
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
    t = 1
    imu_index = 1
    num_measurements = 0
    vo_prev = np.eye(4)
    for i in range(1, len(vo_timestamps)):
        # IMU preintegration
        while imu_index < len(imu_timestamps) and imu_timestamps[imu_index] < vo_timestamps[i]:
            dt = imu_timestamps[imu_index] - imu_timestamps[imu_index-1]
            if not np.isinf(dt) and dt > 0 and dt < 0.5:
                preint_imu.integrateMeasurement(acc_data[imu_index-1], gyro_data[imu_index-1], dt)
                num_measurements += 1
            imu_index += 1
        # thoeretically, we should have at least 3 IMU measurements between each VO frame
        if num_measurements < 3:
            print(f"No IMU data for frame: {i}")
            # if no IMU data between frames, track vo transforms
            vo_prev = vo_prev @ vo_transforms[i]
            continue

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
        
        # add VO factor
        vo_transform = vo_prev @ vo_transforms[i]
        R_vo = vo_transform[:3, :3]
        t_vo = vo_transform[:3, -1]
        pose_vo = gtsam.Pose3(gtsam.Rot3(R_vo), gtsam.Point3(t_vo))
        graph.add(gtsam.BetweenFactorPose3(pose_key(t-1), pose_key(t), pose_vo, vo_noise))

        # add initial values
        initial_values.insert(pose_key(t), identity_pose)
        initial_values.insert(vel_key(t), np.zeros(3))
        initial_values.insert(bias_key(t), initial_bias)

        t += 1
        num_measurements = 0
        vo_prev = np.eye(4)
        preint_imu.resetIntegration()

        if t % mini_batch_size == 0 or i == len(vo_timestamps)-1:
            print("Updating graph...")
            isam.update(graph, initial_values)
            result = isam.calculateEstimate()
            # create new graph for next iteration
            graph = gtsam.NonlinearFactorGraph()
            initial_values = gtsam.Values()

    print("True batch size: ", t)
    
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
    np.random.seed(42)  # For reproducibility
    
    imu_data = np.load('data/imu_data_nik_yellow.npy', allow_pickle=True)
    video_data = np.load('data/video_data_nik_yellow.npy', allow_pickle=True)
    
    refined_transforms = graph_optimization(imu_data, video_data, mini_batch_size=100)
    
    print("Animating trajectory...")
    orientations = refined_transforms[:, :3, :3]
    positions = refined_transforms[:, :3, -1]
    # animate_trajectory(orientations, positions, interval=1)
    plot_trajectory(orientations, positions)
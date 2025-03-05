import Foundation
import CoreMotion
import Network
import AVFoundation

class IMUClient {
    // MARK: - Dependencies
    private let motionManager = CMMotionManager()
    private lazy var tcpClient = TCPClient()
    private var dataTimer: Timer?  // Timer to send data periodically
    
    func connect(to ipAddress: String, with port: UInt16) throws {
        try tcpClient.connect(to: ipAddress, with: port)
    }
    
    func startIMUStreaming() throws {
        guard motionManager.isAccelerometerAvailable,
              motionManager.isGyroAvailable,
              motionManager.isMagnetometerAvailable else {
            print("IMU sensors are not available on this device.")
            return
        }
        
        motionManager.accelerometerUpdateInterval = 0.001  // 1000Hz
        motionManager.gyroUpdateInterval = 0.001
        motionManager.magnetometerUpdateInterval = 0.001
        motionManager.deviceMotionUpdateInterval = 0.001
        
        // Start receiving IMU data
        motionManager.startAccelerometerUpdates()
        motionManager.startGyroUpdates()
        motionManager.startMagnetometerUpdates()
        motionManager.startDeviceMotionUpdates()
        
        // Set up a timer to send data at regular intervals (every 0.1 seconds for example)
        dataTimer = Timer.scheduledTimer(timeInterval: 0.01, target: self, selector: #selector(sendIMUData), userInfo: nil, repeats: true)
    }
    
    @objc func sendIMUData() {
        // Get data from sensors
        if let accData = motionManager.accelerometerData?.acceleration,
           let gyroData = motionManager.gyroData?.rotationRate,
           let magData = motionManager.magnetometerData?.magneticField,
           let quatData = motionManager.deviceMotion?.attitude.quaternion {
            
            let timestamp = CACurrentMediaTime()  // Timestamp in seconds
            let startMarker = Data([0xAB, 0xCD, 0xEF, 0x02])  // 4-byte marker (unique for IMU data)
            let timestampData = withUnsafeBytes(of: timestamp) { Data($0) }
            
            let accXData = withUnsafeBytes(of: accData.x) { Data($0) }
            let accYData = withUnsafeBytes(of: accData.y) { Data($0) }
            let accZData = withUnsafeBytes(of: accData.z) { Data($0) }
            
            let gyroXData = withUnsafeBytes(of: gyroData.x) { Data($0) }
            let gyroYData = withUnsafeBytes(of: gyroData.y) { Data($0) }
            let gyroZData = withUnsafeBytes(of: gyroData.z) { Data($0) }
            
            let magXData = withUnsafeBytes(of: magData.x) { Data($0) }
            let magYData = withUnsafeBytes(of: magData.y) { Data($0) }
            let magZData = withUnsafeBytes(of: magData.z) { Data($0) }
            
            let quatXData = withUnsafeBytes(of: quatData.x) { Data($0) }
            let quatYData = withUnsafeBytes(of: quatData.y) { Data($0) }
            let quatZData = withUnsafeBytes(of: quatData.z) { Data($0) }
            let quatWData = withUnsafeBytes(of: quatData.w) { Data($0) }

            
            
            
            // Combine all data into a single packet
            let packet = startMarker + timestampData +
            accXData + accYData + accZData +
            gyroXData + gyroYData + gyroZData +
            magXData + magYData + magZData +
            quatXData + quatYData + quatZData + quatWData
            
            tcpClient.send(data: packet)
            
            //print("IMU Data Packet: \(packet)")
        } else {
            print("Failed to get IMU data.")
        }
    }
    
    // Call this method to stop streaming when needed
    func stopIMUStreaming() {
        dataTimer?.invalidate()  // Stop the timer
        dataTimer = nil
        
        motionManager.stopAccelerometerUpdates()
        motionManager.stopGyroUpdates()
        motionManager.stopMagnetometerUpdates()
        
        print("Stopped IMU streaming.")
    }
}

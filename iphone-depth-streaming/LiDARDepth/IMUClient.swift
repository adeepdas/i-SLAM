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
        
        motionManager.accelerometerUpdateInterval = 0.005  // 200Hz
        motionManager.gyroUpdateInterval = 0.01
        motionManager.magnetometerUpdateInterval = 0.01
        motionManager.deviceMotionUpdateInterval = 0.01
        
        // Start receiving IMU data
        motionManager.startAccelerometerUpdates()
        motionManager.startGyroUpdates()
        motionManager.startMagnetometerUpdates()
        motionManager.startDeviceMotionUpdates()
        
        
        // Set up a timer to send data at regular intervals (every 0.1 seconds for example)
        dataTimer = Timer.scheduledTimer(timeInterval: 0.01 , target: self, selector: #selector(sendIMUData), userInfo: nil, repeats: true)
    }
    
    @objc func sendIMUData() {
        // Get data from sensors
        if
           let accData = motionManager.deviceMotion?.userAcceleration,
           let rotData = motionManager.deviceMotion?.rotationRate{
            let timestamp = CACurrentMediaTime()  // Timestamp in seconds
            let startMarker = Data([0xAB, 0xCD, 0xEF, 0x02])  // 4-byte marker (unique for IMU data)
            let timestampData = withUnsafeBytes(of: timestamp) { Data($0) }
                        
            let accDataX = withUnsafeBytes(of: accData.x) { Data($0) }
            let accDataY = withUnsafeBytes(of: accData.y) { Data($0) }
            let accDataZ = withUnsafeBytes(of: accData.z) { Data($0) }
            
            let rotDataX = withUnsafeBytes(of: rotData.x) { Data($0) }
            let rotDataY = withUnsafeBytes(of: rotData.y) { Data($0) }
            let rotDataZ = withUnsafeBytes(of: rotData.z) { Data($0) }

            let packet =
            startMarker + timestampData +
            accDataX + accDataY + accDataZ +
            rotDataX + rotDataY + rotDataZ
            tcpClient.send(data: packet)            
        } else {
            print("Failed to get IMU data.")
        }
    }
    
    func buildIMUDataPacket() -> Data? {
        // Safely unwrap all required sensor data.
        guard let rawAccData = motionManager.accelerometerData?.acceleration,
              let rawGyroData = motionManager.gyroData?.rotationRate,
              let rawMagData = motionManager.magnetometerData?.magneticField,
              let accData = motionManager.deviceMotion?.userAcceleration,
              let rotData = motionManager.deviceMotion?.rotationRate,
              let magData = motionManager.deviceMotion?.magneticField,
              let orientationData = motionManager.deviceMotion?.attitude,
              let quatData = motionManager.deviceMotion?.attitude.quaternion
        else {
            print("Failed to get IMU data.")
            return nil
        }
        
        // Create the start marker (4 bytes) and timestamp (8 bytes)
        let startMarker = Data([0xAB, 0xCD, 0xEF, 0x02])  // Unique marker for IMU data
        let timestamp = CACurrentMediaTime()               // Timestamp in seconds (Double)
        let timestampData = withUnsafeBytes(of: timestamp) { Data($0) }
        
        // Convert each sensor value to Data.
        let rawAccX = withUnsafeBytes(of: rawAccData.x) { Data($0) }
        let rawAccY = withUnsafeBytes(of: rawAccData.y) { Data($0) }
        let rawAccZ = withUnsafeBytes(of: rawAccData.z) { Data($0) }
        
        let rawGyroX = withUnsafeBytes(of: rawGyroData.x) { Data($0) }
        let rawGyroY = withUnsafeBytes(of: rawGyroData.y) { Data($0) }
        let rawGyroZ = withUnsafeBytes(of: rawGyroData.z) { Data($0) }
                
        let accDataX = withUnsafeBytes(of: accData.x) { Data($0) }
        let accDataY = withUnsafeBytes(of: accData.y) { Data($0) }
        let accDataZ = withUnsafeBytes(of: accData.z) { Data($0) }
        
        let rotDataX = withUnsafeBytes(of: rotData.x) { Data($0) }
        let rotDataY = withUnsafeBytes(of: rotData.y) { Data($0) }
        let rotDataZ = withUnsafeBytes(of: rotData.z) { Data($0) }
                
        // Combine all the pieces in a specific order.
        let packet = rawAccX + rawAccY + rawAccZ +
                     rawGyroX + rawGyroY + rawGyroZ +
                     accDataX + accDataY + accDataZ +
                     rotDataX + rotDataY + rotDataZ
        
        return packet
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

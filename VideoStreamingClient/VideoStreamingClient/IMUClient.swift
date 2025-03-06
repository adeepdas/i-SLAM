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
        if let rawAccData = motionManager.accelerometerData?.acceleration,
           let rawGyroData = motionManager.gyroData?.rotationRate,
           let rawMagData = motionManager.magnetometerData?.magneticField,
           
           let accData = motionManager.deviceMotion?.userAcceleration,
           let rotData = motionManager.deviceMotion?.rotationRate,
           let magData = motionManager.deviceMotion?.magneticField,
           let orientationData = motionManager.deviceMotion?.attitude,
           let quatData = motionManager.deviceMotion?.attitude.quaternion{
            
            let timestamp = CACurrentMediaTime()  // Timestamp in seconds
            let startMarker = Data([0xAB, 0xCD, 0xEF, 0x02])  // 4-byte marker (unique for IMU data)
            
            let timestampData = withUnsafeBytes(of: timestamp) { Data($0) }
            
            let rawAccX = withUnsafeBytes(of: rawAccData.x) { Data($0) }
            let rawAccY = withUnsafeBytes(of: rawAccData.y) { Data($0) }
            let rawAccZ = withUnsafeBytes(of: rawAccData.z) { Data($0) }
            
            let rawGyroX = withUnsafeBytes(of: rawGyroData.x) { Data($0) }
            let rawGyroY = withUnsafeBytes(of: rawGyroData.y) { Data($0) }
            let rawGyroZ = withUnsafeBytes(of: rawGyroData.z) { Data($0) }
            
            let rawMagX = withUnsafeBytes(of: rawMagData.x) { Data($0) }
            let rawMagY = withUnsafeBytes(of: rawMagData.y) { Data($0) }
            let rawMagZ = withUnsafeBytes(of: rawMagData.z) { Data($0) }
            
            
            
            let accDataX = withUnsafeBytes(of: accData.x) { Data($0) }
            let accDataY = withUnsafeBytes(of: accData.y) { Data($0) }
            let accDataZ = withUnsafeBytes(of: accData.z) { Data($0) }
            
            let rotDataX = withUnsafeBytes(of: rotData.x) { Data($0) }
            let rotDataY = withUnsafeBytes(of: rotData.y) { Data($0) }
            let rotDataZ = withUnsafeBytes(of: rotData.z) { Data($0) }
            
            
            let magDataX = withUnsafeBytes(of: magData.field.x) { Data($0) }
            let magDataY = withUnsafeBytes(of: magData.field.y) { Data($0) }
            let magDataZ = withUnsafeBytes(of: magData.field.z) { Data($0) }
            
                        
            let yaw = withUnsafeBytes(of: orientationData.yaw) { Data($0) }
            let pitch = withUnsafeBytes(of: orientationData.pitch) { Data($0) }
            let roll = withUnsafeBytes(of: orientationData.roll) { Data($0) }

            
            let quatXData = withUnsafeBytes(of: quatData.x) { Data($0) }
            let quatYData = withUnsafeBytes(of: quatData.y) { Data($0) }
            let quatZData = withUnsafeBytes(of: quatData.z) { Data($0) }
            let quatWData = withUnsafeBytes(of: quatData.w) { Data($0) }

            
            
            
            // Combine all data into a single packet
            let packet =
            startMarker + timestampData +
            rawAccX + rawAccY + rawAccZ +
            rawGyroX + rawGyroY + rawGyroZ +
            rawMagX + rawMagY + rawMagZ +
            accDataX + accDataY + accDataZ +
            rotDataX + rotDataY + rotDataZ +
            magDataX + magDataY + magDataZ +
            yaw + pitch + roll +
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

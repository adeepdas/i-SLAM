import AVFoundation
import Foundation
import Network

class VideoClient {
    // MARK: - Dependencies

    private lazy var captureManager = VideoCaptureManager()
    private lazy var videoEncoder = H264Encoder()
    private lazy var tcpClient = TCPClient()

    func connect(to ipAddress: String, with port: UInt16) throws {
        try tcpClient.connect(to: ipAddress, with: port)
    }

    func startSendingVideoToServer() throws {
        try videoEncoder.configureCompressSession()
        captureManager.setVideoOutputDelegate(with: videoEncoder)

        videoEncoder.naluHandling = { [unowned self] data in
            let startMarker = Data([0xAB, 0xCD, 0xEF, 0x01])  // 4-byte marker
            let timestamp = CACurrentMediaTime() // Precise timestamp (seconds)
            let frameLength = UInt32(data.count) // Frame size in bytes
            
            // Convert timestamp and frame length to Data
            let timestampData = withUnsafeBytes(of: timestamp) { Data($0) }
            let lengthData = withUnsafeBytes(of: frameLength) { Data($0) }

            // Create final packet (Start Marker + Timestamp + Frame Size + H.264 Data)
            let packet = startMarker + timestampData + lengthData + data
            
            // Debug: Print first 8 bytes (timestamp)
            // print("Timestamp (Float64):", timestampData.withUnsafeBytes { $0.load(as: Double.self) })
            
            // Send to server
            tcpClient.send(data: packet)
        }
    }
}

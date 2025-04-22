import Foundation
import AVFoundation
import CoreImage
import Network

protocol CaptureDataReceiver: AnyObject {
    func onNewData(capturedData: CameraCapturedData2)
}

class CameraController: NSObject, ObservableObject {
    
    enum ConfigurationError: Error {
        case lidarDeviceUnavailable
        case requiredFormatUnavailable
    }
    
    private let preferredWidthResolution = 1920
    private let preferredHeightResolutuion = 1080
    
    private let videoQueue = DispatchQueue(label: "com.example.apple-samplecode.VideoQueue", qos: .userInteractive)
    
    private(set) var captureSession: AVCaptureSession!
    
    private var photoOutput: AVCapturePhotoOutput!
    private var depthDataOutput: AVCaptureDepthDataOutput!
    private var videoDataOutput: AVCaptureVideoDataOutput!
    private var outputVideoSync: AVCaptureDataOutputSynchronizer!
    
    var allDataClient: TCPClient!
    
    var videoClient: TCPClient!
    var depthClient: TCPClient!
    let imuClient = IMUClient()
    
    lazy var videoEncoder = H264Encoder()
    private var textureCache: CVMetalTextureCache!
    
    weak var delegate: CaptureDataReceiver?
    
    var isFilteringEnabled = true {
        didSet {
            depthDataOutput.isFilteringEnabled = isFilteringEnabled
        }
    }
    
    override init() {
        
        allDataClient = TCPClient()
        videoClient = TCPClient()
        depthClient = TCPClient()
       
        super.init()
        
        do {
            try setupSession()

            
        } catch {
            fatalError("Unable to configure the capture session.")
        }
    }
    
    private func setupSession() throws {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .inputPriority

        // Configure the capture session.
        captureSession.beginConfiguration()
        
        try setupCaptureInput()
        setupCaptureOutputs()
        
        // Finalize the capture session configuration.
        captureSession.commitConfiguration()
    }
    
    private func setupCaptureInput() throws {
        // Look up the LiDAR camera.
        guard let device = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back) else {
            throw ConfigurationError.lidarDeviceUnavailable
        }
        
        // Find a match that outputs video data in the format the app's custom Metal views require.
        guard let format = (device.formats.last { format in
            format.formatDescription.dimensions.width == preferredWidthResolution &&
            format.formatDescription.dimensions.height == preferredHeightResolutuion &&
            format.formatDescription.mediaSubType.rawValue == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange &&
            !format.isVideoBinned &&
            !format.supportedDepthDataFormats.isEmpty
        }) else {
            throw ConfigurationError.requiredFormatUnavailable
        }
        
        // Find a match that outputs depth data in the format the app's custom Metal views require.
        guard let depthFormat = (format.supportedDepthDataFormats.last { depthFormat in
            depthFormat.formatDescription.mediaSubType.rawValue == kCVPixelFormatType_DepthFloat16
        }) else {
            throw ConfigurationError.requiredFormatUnavailable
        }
        
        // Begin the device configuration.
        try device.lockForConfiguration()

        // Configure the device and depth formats.
        device.activeFormat = format
        device.activeDepthDataFormat = depthFormat

        // Finish the device configuration.
        device.unlockForConfiguration()
        
        print("Selected video format: \(device.activeFormat)")
        print("\n\nHeight: \(format.formatDescription.dimensions.height)")
        print("\n\nWidth: \(format.formatDescription.dimensions.width)")
        print("Selected depth format: \(String(describing: device.activeDepthDataFormat))")
        print("\n\nHeight Depth: \(depthFormat.formatDescription.dimensions.height)")
        print("\n\nWidth Depth: \(depthFormat.formatDescription.dimensions.width)")

        // Add a device input to the capture session.
        let deviceInput = try AVCaptureDeviceInput(device: device)
        captureSession.addInput(deviceInput)
    }
    
    private func setupCaptureOutputs() {
        // Create an object to output video sample buffers.
        videoDataOutput = AVCaptureVideoDataOutput()
        captureSession.addOutput(videoDataOutput)
        
        // Create an object to output depth data.
        depthDataOutput = AVCaptureDepthDataOutput()
        depthDataOutput.isFilteringEnabled = isFilteringEnabled
        captureSession.addOutput(depthDataOutput)

        // Create an object to synchronize the delivery of depth and video data.
        outputVideoSync = AVCaptureDataOutputSynchronizer(dataOutputs: [depthDataOutput, videoDataOutput])
        outputVideoSync.setDelegate(self, queue: videoQueue)
        
        // Enable camera intrinsics matrix delivery.
        guard let outputConnection = videoDataOutput.connection(with: .video) else { return }
        if outputConnection.isCameraIntrinsicMatrixDeliverySupported {
            outputConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
        }
        
        // Create an object to output photos.
        photoOutput = AVCapturePhotoOutput()
        photoOutput.maxPhotoQualityPrioritization = .quality
        captureSession.addOutput(photoOutput)

        // Enable delivery of depth data after adding the output to the capture session.
        photoOutput.isDepthDataDeliveryEnabled = true
    }
    
    func startStream() {
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    func stopStream() {
        captureSession.stopRunning()
    }
    
    enum ConnectionError: Error {
        case invalidIPAdress
        case invalidPort
    }

    private lazy var queue = DispatchQueue(label: "tcp.client.queue")
    private var connection: NWConnection?
    private var state: NWConnection.State = .preparing
    
    func sendData(depthData: AVDepthData, cameraIntrinsics: matrix_float3x3) {
        
        self.getEncodedVideoData { encodedPacket in
            self.sendCombinedPacket(videoData: encodedPacket,
                                    depthData: depthData,
                                    intrinsicData: cameraIntrinsics)
        }
    }

    func sendCombinedPacket(videoData: Data, depthData: AVDepthData, intrinsicData: matrix_float3x3)
    {
        let startMarker = Data([0xAB, 0xCD, 0xEF, 0x01])  // 4-byte marker

        let depthBuffer = depthData.depthDataMap
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        
        
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer {CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)}
        let baseAddress = CVPixelBufferGetBaseAddress(depthBuffer)!.assumingMemoryBound(to: Float16.self)
        let bufferSize = width * height * MemoryLayout<Float16>.size
        let depthDataBytes = Data(bytes: baseAddress, count: bufferSize)
        
        let fx: Float = intrinsicData[0][0]  // Focal length in x (pixels)
        let fy: Float = intrinsicData[1][1]  // Focal length in y (pixels)
        let cx: Float = intrinsicData[2][0]   // Principal point x (image center)
        let cy: Float = intrinsicData[2][1]  // Principal point y (image center)
                
        let depthLength = UInt32(depthDataBytes.count) // Frame size in bytes
        
        let timestamp = CACurrentMediaTime() // Precise timestamp (seconds)
        
        let videoLength = UInt32(videoData.count)// Frame size in bytes
            
        let videoLengthBytes = withUnsafeBytes(of: videoLength) {Data($0)}
        let depthLengthBytes = withUnsafeBytes(of: depthLength) {Data($0)}
    
        // Convert timestamp and frame length to Data
        let timestampData = withUnsafeBytes(of: timestamp) { Data($0) }
        let lengthData = withUnsafeBytes(of: videoLength) { Data($0) }
        
        let intrinsicLength = UInt32(16)
        let intrinsicLengthBytes = withUnsafeBytes(of: intrinsicLength) {Data($0)}
        
        let fxBytes = withUnsafeBytes(of: fx) { Data($0) }
        let fyBytes = withUnsafeBytes(of: fy) { Data($0) }
        let cxBytes = withUnsafeBytes(of: cx) { Data($0) }
        let cyBytes = withUnsafeBytes(of: cy) { Data($0) }

        guard let imuPacket = imuClient.buildIMUDataPacket()
        
        else {
            print("IMU data packet is nil; sensor data may not be available.")
            return
        }
        let imuLength = UInt32(imuPacket.count)// Frame size in bytes
        let imuLengthBytes = withUnsafeBytes(of: imuLength) {Data($0)}

        var packet = Data()
        packet.append(startMarker)
        packet.append(timestampData)
        packet.append(videoLengthBytes)
        packet.append(depthLengthBytes)
        packet.append(intrinsicLengthBytes)
        packet.append(videoData)
        packet.append(depthDataBytes)
        packet.append(fxBytes)
        packet.append(fyBytes)
        packet.append(cxBytes)
        packet.append(cyBytes)
        allDataClient.send(data: packet)
    }

    func getEncodedVideoData(completion: @escaping (Data) -> Void) {
        self.videoEncoder.naluHandling = { [unowned self] data in
            let packet = data  // 'data' is the encoded NAL unit
            completion(packet)
        }
    }
    
}

// MARK: Output Synchronizer Delegate
extension CameraController: AVCaptureDataOutputSynchronizerDelegate {
    
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        // Retrieve the synchronized depth and sample buffer container objects.
        guard let syncedDepthData = synchronizedDataCollection.synchronizedData(for: depthDataOutput) as? AVCaptureSynchronizedDepthData,
              let syncedVideoData = synchronizedDataCollection.synchronizedData(for: videoDataOutput) as? AVCaptureSynchronizedSampleBufferData else { return }
        
        
        guard let pixelBuffer = syncedVideoData.sampleBuffer.imageBuffer,
              let cameraCalibrationData = syncedDepthData.depthData.cameraCalibrationData else { return }
        
        
        let data = CameraCapturedData2(depth: syncedDepthData.depthData,
                                        sampleBuffer: syncedVideoData.sampleBuffer,
                                        cameraIntrinsics: cameraCalibrationData.intrinsicMatrix,
                                      cameraReferenceDimensions: cameraCalibrationData.intrinsicMatrixReferenceDimensions)

        delegate?.onNewData(capturedData: data)
    }
}

// MARK: Photo Capture Delegate
extension CameraController: AVCapturePhotoCaptureDelegate {
    
    func capturePhoto() {
        var photoSettings: AVCapturePhotoSettings
        if  photoOutput.availablePhotoPixelFormatTypes.contains(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
            photoSettings = AVCapturePhotoSettings(format: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            ])
        } else {
            photoSettings = AVCapturePhotoSettings()
        }
        
        // Capture depth data with this photo capture.
        photoSettings.isDepthDataDeliveryEnabled = true
        photoOutput.capturePhoto(with: photoSettings, delegate: self)
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        
        // Retrieve the image and depth data.
        guard let pixelBuffer = photo.pixelBuffer,
              let depthData = photo.depthData,
              let cameraCalibrationData = depthData.cameraCalibrationData else { return }

        // Stop the stream until the user returns to streaming mode.
        stopStream()
                
        // Convert the depth data to the expected format.
        let convertedDepth = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat16)
        
        let pixelBuffer2 = convertedDepth.depthDataMap
        CVPixelBufferLockBaseAddress(pixelBuffer2, .readOnly)
                
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        
    
        let data = CameraCapturedData(depth: convertedDepth.depthDataMap.texture(withFormat: .r16Float, planeIndex: 0, addToCache: textureCache),
                                      colorY: pixelBuffer.texture(withFormat: .r8Unorm, planeIndex: 0, addToCache: textureCache),
                                      colorCbCr: pixelBuffer.texture(withFormat: .rg8Unorm, planeIndex: 1, addToCache: textureCache),
                                      cameraIntrinsics: cameraCalibrationData.intrinsicMatrix,
                                      cameraReferenceDimensions: cameraCalibrationData.intrinsicMatrixReferenceDimensions)
    }
}

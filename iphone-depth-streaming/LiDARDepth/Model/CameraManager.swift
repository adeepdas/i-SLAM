/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
An object that connects the camera controller and the views.
*/

import Foundation
import SwiftUI
import Combine
import simd
import AVFoundation
import CoreImage

class CameraManager: ObservableObject, CaptureDataReceiver {
    
    let ipAddress: String
    
    var capturedData: CameraCapturedData
    @Published var isFilteringDepth: Bool {
        didSet {
            controller.isFilteringEnabled = isFilteringDepth
        }
    }
    @Published var orientation = UIDevice.current.orientation
    @Published var waitingForCapture = false
    @Published var processingCapturedResult = false
    @Published var dataAvailable = false
    
    let controller: CameraController
    var cancellables = Set<AnyCancellable>()
    var session: AVCaptureSession { controller.captureSession }
    
    init(ipAddress: String) {
        // Create an object to store the captured data for the views to present.
        self.ipAddress = ipAddress
        capturedData = CameraCapturedData()
        controller = CameraController()
        do {
            try controller.allDataClient.connect(to: self.ipAddress, with: 25005)
            try controller.imuClient.connect(to: self.ipAddress, with: 13005)
            try controller.videoEncoder.configureCompressSession()
            try controller.imuClient.startIMUStreaming()

        } catch {
            print("Error occurred: \(error.localizedDescription)")
        }
        controller.isFilteringEnabled = true
        controller.startStream()
        isFilteringDepth = controller.isFilteringEnabled
        
        NotificationCenter.default.publisher(for: UIDevice.orientationDidChangeNotification).sink { _ in
            self.orientation = UIDevice.current.orientation
        }.store(in: &cancellables)
        controller.delegate = self
    }
    
    func startPhotoCapture() {
        controller.capturePhoto()
        waitingForCapture = true
    }
    
    func resumeStream() {
        controller.startStream()
        processingCapturedResult = false
        waitingForCapture = false
    }
    
    func onNewData(capturedData: CameraCapturedData2) {
        DispatchQueue.main.async {
            if !self.processingCapturedResult {                
                self.controller.videoEncoder.encode(buffer: capturedData.sampleBuffer!)
                self.controller.sendData(depthData: capturedData.depth!, cameraIntrinsics: capturedData.cameraIntrinsics)
                    
                if self.dataAvailable == false {
                    self.dataAvailable = true
                }
            }
        }
    }

}

class CameraCapturedData {
    
    var depth: MTLTexture?
    var colorY: MTLTexture?
    var colorCbCr: MTLTexture?
    var cameraIntrinsics: matrix_float3x3
    var cameraReferenceDimensions: CGSize

    init(depth: MTLTexture? = nil,
         colorY: MTLTexture? = nil,
         colorCbCr: MTLTexture? = nil,
         cameraIntrinsics: matrix_float3x3 = matrix_float3x3(),
         cameraReferenceDimensions: CGSize = .zero) {
        
        self.depth = depth
        self.colorY = colorY
        self.colorCbCr = colorCbCr
        self.cameraIntrinsics = cameraIntrinsics
        self.cameraReferenceDimensions = cameraReferenceDimensions
    }
}

class CameraCapturedData2 {
    
    var depth: AVDepthData?
    var sampleBuffer: CMSampleBuffer?
    var cameraIntrinsics: matrix_float3x3
    var cameraReferenceDimensions: CGSize
    
    init(depth: AVDepthData,
         sampleBuffer: CMSampleBuffer,
         cameraIntrinsics: matrix_float3x3 = matrix_float3x3(),
         cameraReferenceDimensions: CGSize = .zero) {
        
        self.depth = depth
        self.sampleBuffer = sampleBuffer
        self.cameraIntrinsics = cameraIntrinsics
        self.cameraReferenceDimensions = cameraReferenceDimensions
    }
}

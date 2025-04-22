import Foundation
import AVFoundation

extension CVPixelBuffer {
    
    func texture(withFormat pixelFormat: MTLPixelFormat, planeIndex: Int, addToCache cache: CVMetalTextureCache) -> MTLTexture? {
        
        let width = CVPixelBufferGetWidthOfPlane(self, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(self, planeIndex)
        
        var cvtexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, self, nil, pixelFormat, width, height, planeIndex, &cvtexture)
        guard let texture = cvtexture else { return nil }
        return CVMetalTextureGetTexture(texture)
    }
    
}

extension CMSampleBuffer {
    var isKeyFrame: Bool {
        let attachments = CMSampleBufferGetSampleAttachmentsArray(self, createIfNecessary: true) as? [[CFString: Any]]

        let isNotKeyFrame = (attachments?.first?[kCMSampleAttachmentKey_NotSync] as? Bool) ?? false

        return !isNotKeyFrame
    }
}

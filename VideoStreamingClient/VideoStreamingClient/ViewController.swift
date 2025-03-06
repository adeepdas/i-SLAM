//
//  ViewController.swift
//  VideoStreamingClient
//
//  Created by Jade on 2022/09/20.
//

import UIKit

class ViewController: UIViewController {
    @IBOutlet var addressTextFiled: UITextField!

    let videoClient = VideoClient()
    let imuClient = IMUClient()

    override func viewDidLoad() {
        super.viewDidLoad()
    }

    @IBAction func startButtonTapped(_: UIButton) {
        let defaultIPAddress = "192.168.5.243"
        
        let ipAddress = addressTextFiled.text?.trimmingCharacters(in: .whitespacesAndNewlines)
        
        let finalIPAddress = (ipAddress?.isEmpty ?? true) ? defaultIPAddress : ipAddress!
        
        do {
            try videoClient.connect(to: finalIPAddress, with: 12005)
            try imuClient.connect(to: finalIPAddress, with: 13005)
            try videoClient.startSendingVideoToServer()
            try imuClient.startIMUStreaming()
        } catch {
            print("Error occurred: \(error.localizedDescription)")
        }
    }
}

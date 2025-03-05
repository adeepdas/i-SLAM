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
        //guard let ipAddress = addressTextFiled.text else { return }
        let ipAddress = "192.168.5.243"
        
        do {
            try videoClient.connect(to: ipAddress, with: 12005)
            try imuClient.connect(to: ipAddress, with: 13005)
            try videoClient.startSendingVideoToServer()
            try imuClient.startIMUStreaming()
        } catch {
            print("error occured : \(error.localizedDescription)")
        }
    }
}

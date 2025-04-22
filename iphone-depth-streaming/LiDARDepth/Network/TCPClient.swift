//
//  TCPClient.swift
//  CameraTest
//
//  Created by Jade on 2022/09/16.
//

import Foundation
import Network

class TCPClient {
    enum ConnectionError: Error {
        case invalidIPAdress
        case invalidPort
    }

    private lazy var queue = DispatchQueue(label: "tcp.client.queue")
    private var connection: NWConnection?
    private var state: NWConnection.State = .preparing

    func connect(to ipAddress: String, with port: UInt16) throws {
        guard let ipAddress = IPv4Address(ipAddress) else {
            throw ConnectionError.invalidIPAdress
        }
        guard let port = NWEndpoint.Port(rawValue: port) else {
            throw ConnectionError.invalidPort
        }
        let host = NWEndpoint.Host.ipv4(ipAddress)

        connection = NWConnection(host: host, port: port, using: .tcp)

        connection?.stateUpdateHandler = { [unowned self] state in
            self.state = state
        }

        connection?.start(queue: queue)
    }

    func send(data: Data) {
        guard state == .ready else { return }

        connection?.send(content: data,
                         completion: .contentProcessed { error in
                             if let error = error {
                                 print(error)
                             }
                         })
    }
}

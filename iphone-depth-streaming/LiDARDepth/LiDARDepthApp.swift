/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The sample's main app.
*/

import SwiftUI

@main
struct LiDARDepthApp: App {
    // Create a state variable to store the server IP address
    @State private var serverIPAddress: String = "35.3.202.167" // Provide a default IP

    var body: some Scene {
        WindowGroup {
            // Pass the @State variable as a @Binding to ContentView
            ContentView(serverIPAddress: $serverIPAddress)
        }
    }
}

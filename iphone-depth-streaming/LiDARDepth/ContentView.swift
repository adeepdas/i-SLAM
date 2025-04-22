import SwiftUI
import MetalKit
import Metal

struct ContentView: View {
   @State var manager: CameraManager? = nil

    @Binding var serverIPAddress: String
    @State private var isConnected: Bool = false

    @State private var maxDepth = Float(5.0)
    @State private var minDepth = Float(0.0)  

    let maxRangeDepth = Float(15)
    let minRangeDepth = Float(0)

    var body: some View {
        VStack {
            // IP Address Input + Connect Button
            HStack {
                Text("Server IP:")
                TextField("Enter IP address", text: $serverIPAddress)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .keyboardType(.numbersAndPunctuation)
                    .frame(width: 200)

                Button("Connect") {
                    manager = CameraManager(ipAddress: serverIPAddress)
                    isConnected = true
                }
                .buttonStyle(.borderedProminent)
                .disabled(serverIPAddress.isEmpty || isConnected)
            }
            .padding(.vertical, 10)

            // Show controls only when connected
            if let manager = manager, isConnected {

                SliderDepthBoundaryView(val: $maxDepth, label: "Max Depth", minVal: minRangeDepth, maxVal: maxRangeDepth)
                SliderDepthBoundaryView(val: $minDepth, label: "Min Depth", minVal: minRangeDepth, maxVal: maxRangeDepth)

                ScrollView {
                    LazyVGrid(columns: [GridItem(.flexible(maximum: 600)), GridItem(.flexible(maximum: 600))]) {
                    }
                }
            } else {
                Text("Please connect to a server first.")
                    .foregroundColor(.gray)
                    .padding()
            }
        }
        .padding()
    }
}

struct SliderDepthBoundaryView: View {
    @Binding var val: Float
    var label: String
    var minVal: Float
    var maxVal: Float
    let stepsCount = Float(200.0)

    var body: some View {
        HStack {
            Text(String(format: " %@: %.2f", label, val))
            Slider(
                value: $val,
                in: minVal...maxVal,
                step: (maxVal - minVal) / stepsCount
            ) {
            } minimumValueLabel: {
                Text(String(format: "%.1f", minVal))
            } maximumValueLabel: {
                Text(String(format: "%.1f", maxVal))
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    @State static var serverIPAddress = "35.3.202.167" // Create a mock server IP address for preview

    static var previews: some View {
        ContentView(serverIPAddress: $serverIPAddress)
            .previewDevice("iPhone 12 Pro Max")
    }
}

import Foundation
import Combine

#if canImport(ActivityKit)
import ActivityKit

struct AdvisorSessionAttributes: ActivityAttributes {
    struct ContentState: Codable, Hashable {
        var phase: String
        var health: String
        var openPositionCount: Int
        var latestEvent: String
        var updatedAt: Date
    }

    var sessionDate: String
    var mode: String
}
#endif

@MainActor
final class LiveActivityManager: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var statusMessage = "Not started"

#if canImport(ActivityKit)
    private var activity: Activity<AdvisorSessionAttributes>?
#endif

    func startDemoSession(openPositions: Int = 1) async {
#if canImport(ActivityKit)
        guard ActivityAuthorizationInfo().areActivitiesEnabled else {
            statusMessage = "Live Activities are disabled on this device"
            return
        }

        let attributes = AdvisorSessionAttributes(
            sessionDate: ISO8601DateFormatter().string(from: Date()),
            mode: "paper",
        )
        let state = AdvisorSessionAttributes.ContentState(
            phase: "Monitoring",
            health: "Healthy",
            openPositionCount: openPositions,
            latestEvent: "Demo session started",
            updatedAt: Date(),
        )

        do {
            let content = ActivityContent(
                state: state,
                staleDate: Date().addingTimeInterval(3_600),
            )
            activity = try Activity.request(attributes: attributes, content: content, pushType: nil)
            isRunning = true
            statusMessage = "Session activity running"
        } catch {
            statusMessage = "Unavailable: \(error.localizedDescription)"
        }
#else
        statusMessage = "ActivityKit is unavailable in this build"
#endif
    }

    func endDemoSession() async {
#if canImport(ActivityKit)
        guard let activity else {
            statusMessage = "No active session"
            return
        }
        let finalState = AdvisorSessionAttributes.ContentState(
            phase: "Ended",
            health: "Complete",
            openPositionCount: 0,
            latestEvent: "Session ended",
            updatedAt: Date(),
        )
        await activity.end(
            ActivityContent(state: finalState, staleDate: nil),
            dismissalPolicy: .after(Date().addingTimeInterval(900)),
        )
        self.activity = nil
        isRunning = false
        statusMessage = "Session ended"
#else
        statusMessage = "ActivityKit is unavailable in this build"
#endif
    }
}

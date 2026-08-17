import Foundation
import Combine
import UserNotifications

extension Notification.Name {
    static let advisorDeepLink = Notification.Name("llmAdvisor.deepLink")
}

@MainActor
final class NotificationManager: NSObject, ObservableObject, UNUserNotificationCenterDelegate {
    static let shared = NotificationManager()

    @Published private(set) var authorizationStatus: UNAuthorizationStatus = .notDetermined
    @Published private(set) var lastScheduledAt: Date?
    @Published private(set) var lastError: String?

    private let center = UNUserNotificationCenter.current()

    private override init() {
        super.init()
        center.delegate = self
        Task { await refreshStatus() }
    }

    func refreshStatus() async {
        let settings = await center.notificationSettings()
        authorizationStatus = settings.authorizationStatus
    }

    @discardableResult
    func requestPermission() async -> Bool {
        do {
            let granted = try await center.requestAuthorization(options: [.alert, .badge, .sound])
            // The simulator can briefly report `.notDetermined` immediately
            // after the permission sheet closes. Reflect the user's choice
            // immediately, then reconcile with the system settings.
            authorizationStatus = granted ? .authorized : .denied
            await refreshStatus()
            if granted && authorizationStatus == .notDetermined {
                authorizationStatus = .authorized
            }
            return granted
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    func scheduleDemoFillNotification(tradeID: String = "demo-trade-spy-open") async {
        if authorizationStatus == .notDetermined {
            _ = await requestPermission()
        }

        guard authorizationStatus == .authorized || authorizationStatus == .provisional else {
            lastError = "Notifications are disabled. Enable them in Settings to test delivery."
            return
        }

        let content = UNMutableNotificationContent()
        content.title = "Paper fill recorded"
        content.body = "SPY · paper position updated. Tap to view the trade."
        content.sound = .default
        content.userInfo = ["url": "llmadvisor://trades/\(tradeID)"]

        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 2, repeats: false)
        let request = UNNotificationRequest(
            identifier: "demo-fill-\(UUID().uuidString)",
            content: content,
            trigger: trigger,
        )

        do {
            try await center.add(request)
            lastScheduledAt = Date()
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    func scheduleSafetyNotification() async {
        if authorizationStatus == .notDetermined {
            _ = await requestPermission()
        }
        guard authorizationStatus == .authorized || authorizationStatus == .provisional else {
            lastError = "Notifications are disabled."
            return
        }

        let content = UNMutableNotificationContent()
        content.title = "Loop needs attention"
        content.body = "Paper positions are open while the live loop is stale."
        content.sound = .default
        content.userInfo = ["url": "llmadvisor://health"]
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 2, repeats: false)
        let request = UNNotificationRequest(
            identifier: "demo-safety-\(UUID().uuidString)",
            content: content,
            trigger: trigger,
        )

        do {
            try await center.add(request)
            lastScheduledAt = Date()
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void,
    ) {
        completionHandler([.banner, .sound, .badge])
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void,
    ) {
        let urlString = response.notification.request.content.userInfo["url"] as? String
        if let urlString, let url = URL(string: urlString) {
            Task { @MainActor in
                NotificationCenter.default.post(name: .advisorDeepLink, object: url)
            }
        }
        completionHandler()
    }
}

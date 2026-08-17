import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var store: AppStore
    @State private var notificationsEnabled = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Session") {
                    LabeledContent("Account", value: store.provider.rawValue)
                    LabeledContent("Connection", value: store.networkState.rawValue)
                    LabeledContent("Last refresh", value: advisorRelativeTime(store.lastRefreshAt))
                    Button("Sign out", role: .destructive) {
                        store.signOut()
                    }
                }

                Section {
                    Toggle("High-signal alerts", isOn: $notificationsEnabled)
                        .onChange(of: notificationsEnabled) { _, enabled in
                            guard enabled else { return }
                            Task { _ = await store.notifications.requestPermission() }
                        }

                    LabeledContent("Authorization", value: notificationStatus)

                    Button {
                        Task { await store.notifications.scheduleDemoFillNotification() }
                    } label: {
                        Label("Send test fill alert", systemImage: "bell.badge")
                    }
                    .disabled(!notificationsEnabled)

                    Button {
                        Task { await store.notifications.scheduleSafetyNotification() }
                    } label: {
                        Label("Send loop safety alert", systemImage: "exclamationmark.triangle")
                    }
                    .disabled(!notificationsEnabled)

                    if let scheduled = store.notifications.lastScheduledAt {
                        Text("Last test scheduled \(advisorRelativeTime(scheduled))")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    if let error = store.notifications.lastError {
                        Text(error)
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                } header: {
                    Text("Notifications")
                } footer: {
                    Text("Alerts are intentionally high-signal. They open directly into the relevant native screen.")
                }

                Section {
                    LabeledContent("Status", value: store.liveActivities.statusMessage)
                    Button {
                        Task { await store.liveActivities.startDemoSession(openPositions: store.snapshot.positions.count) }
                    } label: {
                        Label("Start demo session", systemImage: "dot.radiowaves.left.and.right")
                    }
                    .disabled(store.liveActivities.isRunning)

                    Button {
                        Task { await store.liveActivities.endDemoSession() }
                    } label: {
                        Label("End demo session", systemImage: "stop.circle")
                    }
                    .disabled(!store.liveActivities.isRunning)
                } header: {
                    Text("Live Activity")
                } footer: {
                    Text("The session is a private, discreet monitoring surface. It never displays symbols, dollars, or P&L on the Lock Screen.")
                }

                Section("Demo and diagnostics") {
                    Toggle("Simulate offline", isOn: $store.simulateOffline)
                    Button {
                        Task { await store.refresh() }
                    } label: {
                        Label("Refresh now", systemImage: "arrow.clockwise")
                    }
                    Button {
                        if let id = store.latestTradeID() {
                            store.openTrade(id)
                        }
                    } label: {
                        Label("Open latest trade", systemImage: "arrow.up.right.square")
                    }
                }

                Section("About") {
                    LabeledContent("App", value: "LLM Advisor")
                    LabeledContent("Build", value: "Native iOS · iOS 18+")
                    Text("Read-only companion for the server-side paper loop. Broker credentials and execution remain outside the app.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .task {
                await store.notifications.refreshStatus()
                notificationsEnabled = store.notifications.authorizationStatus == .authorized || store.notifications.authorizationStatus == .provisional
            }
        }
    }

    private var notificationStatus: String {
        switch store.notifications.authorizationStatus {
        case .authorized: return "Allowed"
        case .provisional: return "Provisional"
        case .denied: return "Disabled"
        case .ephemeral: return "Temporary"
        case .notDetermined: return "Not requested"
        @unknown default: return "Unknown"
        }
    }
}

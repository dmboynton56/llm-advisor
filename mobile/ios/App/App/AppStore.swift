import Foundation
import Combine
import SwiftUI

@MainActor
final class AppStore: ObservableObject {
    @Published private(set) var snapshot = DemoData.snapshot
    @Published private(set) var trades = DemoData.trades
    @Published private(set) var isAuthenticated: Bool
    @Published private(set) var provider: SessionProvider = .demo
    @Published private(set) var networkState: AppNetworkState = .demo
    @Published private(set) var isRefreshing = false
    @Published private(set) var isPairing = false
    @Published private(set) var lastError: String?
    @Published private(set) var lastRefreshAt = Date()
    @Published var selectedTab: AppTab = .today
    @Published var pendingTradeID: String?
    @Published var shouldShowHealth = false
    @Published var simulateOffline = false

    let notifications = NotificationManager.shared
    let liveActivities = LiveActivityManager()
    private let apiClient = MobileAPIClient()

    private let demoSessionKey = "llm-advisor.demo-session"
    private let paperTokenKey = "llm-advisor.private-paper-token"
    private let paperExpiryKey = "llm-advisor.private-paper-token-expiry"
    private var paperAccessToken: String?
    private var deepLinkObserver: NSObjectProtocol?

    init() {
        paperAccessToken = KeychainStore.read(key: paperTokenKey)
        if paperAccessToken != nil {
            isAuthenticated = true
            provider = .paper
            networkState = .live
        } else {
            isAuthenticated = UserDefaults.standard.bool(forKey: demoSessionKey)
        }
        deepLinkObserver = NotificationCenter.default.addObserver(
            forName: .advisorDeepLink,
            object: nil,
            queue: .main,
        ) { [weak self] notification in
            guard let url = notification.object as? URL else { return }
            Task { @MainActor [weak self] in
                self?.handleDeepLink(url)
            }
        }
        if paperAccessToken != nil {
            Task { @MainActor [weak self] in
                await self?.restorePaperSession()
            }
        }
    }

    deinit {
        if let deepLinkObserver {
            NotificationCenter.default.removeObserver(deepLinkObserver)
        }
    }

    func continueWithDemo() {
        clearPaperSession()
        provider = .demo
        networkState = .demo
        isAuthenticated = true
        UserDefaults.standard.set(true, forKey: demoSessionKey)
        lastError = nil
    }

    func completeAppleSignIn(identityToken: String, nonce: String?) async {
        do {
            let session = try await apiClient.exchangeAppleIdentityToken(identityToken, nonce: nonce)
            let bootstrap = try await apiClient.fetchBootstrap(accessToken: session.accessToken)
            snapshot = DashboardSnapshot(
                account: bootstrap.account,
                health: bootstrap.health,
                equityHistory: bootstrap.equityHistory,
                positions: bootstrap.positions,
                latestDecision: bootstrap.latestDecision,
                performance: bootstrap.performance,
                breakdowns: bootstrap.breakdowns,
                funnel: bootstrap.funnel,
                brokerOrders: [],
                generatedAt: bootstrap.generatedAt,
            )
            trades = bootstrap.trades
            provider = .apple
            networkState = .live
            isAuthenticated = true
            // The demo flag is deliberately not used to persist a real Apple
            // session. A production build will move the access/refresh tokens
            // into Keychain before allowing an authenticated relaunch.
            UserDefaults.standard.set(false, forKey: demoSessionKey)
            lastRefreshAt = Date()
            lastError = nil
        } catch {
            lastError = "Apple sign-in could not reach the private mobile API. \(error.localizedDescription)"
        }
    }

    func pairPrivatePaperAccount(code: String) async {
        guard !isPairing else { return }
        isPairing = true
        defer { isPairing = false }

        do {
            let session = try await apiClient.pairPrivatePaperAccount(code: code)
            guard session.provider == "alpaca", session.environment == "paper", session.readOnly else {
                throw MobileAPIClient.ClientError.invalidResponse
            }
            try await hydratePaperSession(accessToken: session.accessToken)
            paperAccessToken = session.accessToken
            KeychainStore.save(session.accessToken, key: paperTokenKey)
            UserDefaults.standard.set(session.expiresAt.timeIntervalSince1970, forKey: paperExpiryKey)
            UserDefaults.standard.set(false, forKey: demoSessionKey)
            provider = .paper
            networkState = .live
            isAuthenticated = true
            lastError = nil
        } catch let error as MobileAPIClient.ClientError {
            lastError = error == .unauthorized
                ? "That private pairing code was not accepted."
                : "The private paper account could not be reached."
        } catch {
            lastError = "The private paper account could not be reached. \(error.localizedDescription)"
        }
    }

    func signOut() {
        clearPaperSession()
        isAuthenticated = false
        provider = .demo
        networkState = .demo
        UserDefaults.standard.set(false, forKey: demoSessionKey)
        selectedTab = .today
        pendingTradeID = nil
    }

    func refresh() async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        if simulateOffline {
            networkState = .offline
            lastError = "Couldn’t refresh right now. Showing the last saved reading."
            return
        }

        if provider == .paper, let paperAccessToken {
            do {
                try await hydratePaperSession(accessToken: paperAccessToken)
                networkState = .live
                lastError = nil
            } catch let error as MobileAPIClient.ClientError {
                if error == .unauthorized {
                    signOut()
                    lastError = "The private paper session expired. Pair the account again."
                } else {
                    networkState = .offline
                    lastError = "Couldn’t refresh the paper account. Showing the last saved reading."
                }
            } catch {
                networkState = .offline
                lastError = "Couldn’t refresh the paper account. Showing the last saved reading."
            }
            return
        }

        try? await Task.sleep(nanoseconds: 280_000_000)
        networkState = provider == .apple ? .live : .demo
        lastRefreshAt = Date()
        lastError = nil
        snapshot.generatedAt = Date()
        snapshot.health.heartbeat = Date().addingTimeInterval(-12)
    }

    func openTrade(_ id: String) {
        selectedTab = .trades
        pendingTradeID = id
    }

    func handleDeepLink(_ url: URL) {
        let host = url.host?.lowercased() ?? ""
        let components = url.pathComponents.filter { $0 != "/" }
        let route = host.isEmpty ? components.first?.lowercased() ?? "today" : host

        switch route {
        case "trades", "trade":
            selectedTab = .trades
            if let id = components.last, id != "trades", id != "trade" {
                pendingTradeID = id
            }
        case "insights", "breakdowns", "funnel":
            selectedTab = .insights
        case "settings":
            selectedTab = .settings
        case "health", "today":
            selectedTab = .today
            shouldShowHealth = route == "health"
        default:
            selectedTab = .today
        }
    }

    func latestTradeID() -> String? {
        trades.first?.id
    }

    private func restorePaperSession() async {
        guard let paperAccessToken else { return }
        let expiry = UserDefaults.standard.double(forKey: paperExpiryKey)
        if expiry > 0 && expiry <= Date().timeIntervalSince1970 {
            signOut()
            return
        }
        do {
            try await hydratePaperSession(accessToken: paperAccessToken)
            provider = .paper
            networkState = .live
            isAuthenticated = true
            lastError = nil
        } catch let error as MobileAPIClient.ClientError {
            if error == .unauthorized {
                signOut()
            } else {
                lastError = "The private paper account could not be refreshed."
            }
        } catch {
            lastError = "The private paper account could not be refreshed."
        }
    }

    private func hydratePaperSession(accessToken: String) async throws {
        let bootstrap = try? await apiClient.fetchBootstrap(accessToken: accessToken)
        let alpaca = try await apiClient.fetchAlpacaPaper(accessToken: accessToken)

        if let bootstrap {
            snapshot = DashboardSnapshot(
                account: bootstrap.account,
                health: bootstrap.health,
                equityHistory: bootstrap.equityHistory,
                positions: bootstrap.positions,
                latestDecision: bootstrap.latestDecision,
                performance: bootstrap.performance,
                breakdowns: bootstrap.breakdowns,
                funnel: bootstrap.funnel,
                brokerOrders: snapshot.brokerOrders,
                generatedAt: bootstrap.generatedAt,
            )
            trades = bootstrap.trades
        }

        let positions = alpaca.positions.map { position in
            PaperPosition(
                id: "alpaca-\(position.symbol)",
                symbol: position.underlyingSymbol ?? position.symbol,
                optionSymbol: position.optionSymbol ?? position.symbol,
                side: position.side,
                quantity: position.quantity,
                entryPrice: position.entryPrice,
                currentPrice: position.currentPrice,
                unrealizedPnl: position.unrealizedPnl,
                unrealizedPnlPercent: position.unrealizedPnlPercent,
                setup: position.setup ?? "Alpaca paper position",
                dte: position.dte,
                openedAt: position.openedAt,
                stopMark: position.stopMark,
                targetMark: position.targetMark,
            )
        }
        let orders = (alpaca.openOrders + alpaca.recentOrders).reduce(into: [PaperOrder]()) { result, order in
            let id = order.id ?? "\(order.symbol)-\(order.submittedAt?.timeIntervalSince1970 ?? Date().timeIntervalSince1970)"
            guard !result.contains(where: { $0.id == id }) else { return }
            result.append(
                PaperOrder(
                    id: id,
                    symbol: order.symbol,
                    side: order.side,
                    type: order.type,
                    quantity: order.quantity,
                    filledQuantity: order.filledQuantity,
                    limitPrice: order.limitPrice,
                    stopPrice: order.stopPrice,
                    filledAveragePrice: order.filledAveragePrice,
                    status: order.status,
                    submittedAt: order.submittedAt,
                    filledAt: order.filledAt,
                )
            )
        }

        snapshot.account = AccountSummary(
            equity: alpaca.account.equity,
            dailyPnl: alpaca.account.dailyPnl,
            dailyPnlPercent: alpaca.account.dailyPnlPercent,
            buyingPower: alpaca.account.buyingPower,
        )
        snapshot.positions = positions
        snapshot.brokerOrders = orders
        if let equity = alpaca.account.equity {
            let point = EquityPoint(
                capturedAt: alpaca.fetchedAt,
                equity: equity,
                dailyPnl: alpaca.account.dailyPnl,
            )
            snapshot.equityHistory = Array((snapshot.equityHistory + [point]).suffix(90))
        }
        snapshot.generatedAt = alpaca.fetchedAt
        lastRefreshAt = alpaca.fetchedAt
    }

    private func clearPaperSession() {
        paperAccessToken = nil
        KeychainStore.delete(key: paperTokenKey)
        UserDefaults.standard.removeObject(forKey: paperExpiryKey)
    }
}

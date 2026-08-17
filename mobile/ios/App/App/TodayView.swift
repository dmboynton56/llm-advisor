import SwiftUI

struct TodayView: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        NavigationStack {
            ScrollView(showsIndicators: false) {
                LazyVStack(alignment: .leading, spacing: 18) {
                    header
                    if let error = store.lastError {
                        InlineStatusBanner(text: error, tone: .warning)
                    }
                    healthCard
                    accountCard
                    positionsCard
                    decisionCard
                    quickActions
                    lastUpdated
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 28)
            }
            .background(AdvisorTheme.paper.ignoresSafeArea())
            .navigationTitle("Today")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await store.refresh() }
                    } label: {
                        Image(systemName: store.isRefreshing ? "arrow.triangle.2.circlepath" : "arrow.clockwise")
                    }
                    .accessibilityLabel("Refresh dashboard")
                    .symbolEffect(.rotate, isActive: store.isRefreshing)
                }
            }
            .refreshable {
                await store.refresh()
            }
            .sheet(isPresented: $store.shouldShowHealth) {
                HealthDetailView()
                    .environmentObject(store)
                    .presentationDetents([.medium, .large])
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            Image("LLMAdvisorMark")
                .resizable()
                .scaledToFill()
                .frame(width: 42, height: 42)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 3) {
                Text("LLM ADVISOR")
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .tracking(1.4)
                    .foregroundStyle(AdvisorTheme.muted)
                Text("Pocket briefing")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundStyle(AdvisorTheme.ink)
            }

            Spacer()

            Image(systemName: "moon.circle")
                .font(.system(size: 27, weight: .light))
                .foregroundStyle(AdvisorTheme.muted)
                .accessibilityLabel("System appearance")
        }
        .padding(.top, 2)
    }

    private var healthCard: some View {
        Button {
            store.shouldShowHealth = true
        } label: {
            HStack(spacing: 14) {
                ZStack {
                    Circle()
                        .stroke((store.snapshot.health.isHealthy ? AdvisorTheme.gain : AdvisorTheme.loss).opacity(0.18), lineWidth: 7)
                        .frame(width: 40, height: 40)
                    Circle()
                        .fill(store.snapshot.health.isHealthy ? AdvisorTheme.gain : AdvisorTheme.loss)
                        .frame(width: 12, height: 12)
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(store.snapshot.health.isHealthy ? "LIVE LOOP HEALTHY" : "LIVE LOOP NEEDS ATTENTION")
                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                        .tracking(0.7)
                        .foregroundStyle(store.snapshot.health.isHealthy ? AdvisorTheme.gain : AdvisorTheme.loss)
                    Text("\(store.snapshot.health.message) · \(store.snapshot.health.symbolsTracked ?? 0) symbols")
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(AdvisorTheme.muted)
                }

                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(AdvisorTheme.muted)
            }
            .padding(16)
            .advisorCard()
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Live loop health")
        .accessibilityValue(store.snapshot.health.isHealthy ? "Healthy" : "Needs attention")
        .accessibilityHint("Opens health details")
    }

    private var accountCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardLabel(text: "PAPER ACCOUNT EQUITY")
            Text(store.snapshot.account.equity.advisorCurrency)
                .font(.system(size: 38, weight: .medium, design: .monospaced))
                .tracking(-1.5)
                .foregroundStyle(AdvisorTheme.ink)
                .minimumScaleFactor(0.72)

            HStack(spacing: 12) {
                Text(store.snapshot.account.dailyPnl.advisorSignedCurrency)
                    .foregroundStyle((store.snapshot.account.dailyPnl ?? 0) >= 0 ? AdvisorTheme.gain : AdvisorTheme.loss)
                Text("\(store.snapshot.account.dailyPnlPercent.advisorSignedPercent) today")
                    .foregroundStyle(AdvisorTheme.muted)
            }
            .font(.system(size: 14, weight: .medium, design: .monospaced))

            AdvisorEquityChart(points: store.snapshot.equityHistory)
                .frame(height: 112)

            Text("\(store.networkState.rawValue) · \(advisorRelativeTime(store.snapshot.generatedAt))")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(AdvisorTheme.muted)
        }
        .padding(20)
        .advisorCard(radius: 22)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Paper account equity")
        .accessibilityValue("\(store.snapshot.account.equity.advisorCurrency), \(store.snapshot.account.dailyPnl.advisorSignedCurrency) today")
    }

    private var positionsCard: some View {
        VStack(alignment: .leading, spacing: 11) {
            SectionHeader(title: "OPEN POSITIONS", detail: "\(store.snapshot.positions.count) open")

            if store.snapshot.positions.isEmpty {
                EmptyCard(text: "Flat — no open positions.")
            } else {
                ForEach(store.snapshot.positions) { position in
                    Button {
                        if let trade = store.trades.first(where: { $0.symbol == position.symbol && $0.status == "Open" }) {
                            store.openTrade(trade.id)
                        } else if let id = store.latestTradeID() {
                            store.openTrade(id)
                        }
                    } label: {
                        HStack(spacing: 14) {
                            VStack(alignment: .leading, spacing: 5) {
                                Text(position.symbol)
                                    .font(.system(size: 18, weight: .bold, design: .monospaced))
                                    .foregroundStyle(AdvisorTheme.ink)
                                Text("\(position.setup ?? "Paper position") · \(Int(position.quantity)) contracts")
                                    .font(.system(size: 12, design: .monospaced))
                                    .foregroundStyle(AdvisorTheme.muted)
                            }
                            Spacer()
                            VStack(alignment: .trailing, spacing: 5) {
                                Text(position.unrealizedPnl.advisorSignedCurrency)
                                    .font(.system(size: 16, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(position.unrealizedPnl >= 0 ? AdvisorTheme.gain : AdvisorTheme.loss)
                                Text("\(position.unrealizedPnlPercent.advisorPercent) open P&L")
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(AdvisorTheme.muted)
                            }
                            Image(systemName: "chevron.right")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(AdvisorTheme.muted)
                        }
                        .padding(16)
                        .advisorCard(radius: 16)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Open position \(position.symbol)")
                    .accessibilityValue(position.unrealizedPnl.advisorSignedCurrency)
                }
            }
        }
    }

    private var decisionCard: some View {
        VStack(alignment: .leading, spacing: 11) {
            SectionHeader(title: "LATEST DECISION", detail: store.snapshot.latestDecision?.confidence.advisorPercent ?? "—")
            if let decision = store.snapshot.latestDecision {
                Button {
                    if let id = store.trades.first(where: { $0.symbol == decision.symbol })?.id {
                        store.openTrade(id)
                    }
                } label: {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("\(decision.symbol) · \(decision.setup ?? "DECISION")")
                                .font(.system(size: 13, weight: .bold, design: .monospaced))
                                .foregroundStyle(AdvisorTheme.ink)
                            Spacer()
                            Text(decision.verdict.uppercased())
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .tracking(0.8)
                                .foregroundStyle(AdvisorTheme.ink)
                                .padding(.horizontal, 9)
                                .padding(.vertical, 5)
                                .background(AdvisorTheme.sunk)
                                .clipShape(Capsule())
                        }
                        Text(decision.reasoning ?? "No reasoning captured.")
                            .font(.system(size: 13, design: .rounded))
                            .foregroundStyle(AdvisorTheme.muted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(16)
                    .advisorCard(radius: 16)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Latest decision for \(decision.symbol)")
                .accessibilityValue("\(decision.verdict), \(decision.confidence.advisorPercent) confidence")
            } else {
                EmptyCard(text: "No recent decision recorded.")
            }
        }
    }

    private var quickActions: some View {
        VStack(alignment: .leading, spacing: 11) {
            SectionHeader(title: "QUICK ACTIONS", detail: "Read-only")
            HStack(spacing: 10) {
                QuickActionButton(title: "Trades", symbol: "arrow.left.arrow.right") {
                    store.selectedTab = .trades
                }
                QuickActionButton(title: "Insights", symbol: "chart.xyaxis.line") {
                    store.selectedTab = .insights
                }
                QuickActionButton(title: "Test alert", symbol: "bell.badge") {
                    Task { await store.notifications.scheduleDemoFillNotification() }
                }
            }
        }
    }

    private var lastUpdated: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(store.networkState == .offline ? AdvisorTheme.loss : AdvisorTheme.gain)
                .frame(width: 7, height: 7)
            Text("Updated \(advisorRelativeTime(store.lastRefreshAt))")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(AdvisorTheme.muted)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.top, 3)
    }
}

struct HealthDetailView: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        NavigationStack {
            List {
                Section("Current status") {
                    LabeledContent("Loop", value: store.snapshot.health.isHealthy ? "Healthy" : "Needs attention")
                    LabeledContent("Last heartbeat", value: advisorRelativeTime(store.snapshot.health.heartbeat))
                    LabeledContent("Loop count", value: "\(store.snapshot.health.loopCount ?? 0)")
                    LabeledContent("Symbols", value: "\(store.snapshot.health.symbolsTracked ?? 0)")
                }
                Section("Safety") {
                    Label("No live-money execution is available in this app.", systemImage: "checkmark.shield")
                    Label("The paper loop remains server-side.", systemImage: "server.rack")
                }
            }
            .navigationTitle("Loop health")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

private struct AdvisorEquityChart: View {
    let points: [EquityPoint]

    var body: some View {
        GeometryReader { proxy in
            let minValue = points.map(\.equity).min() ?? 0
            let maxValue = points.map(\.equity).max() ?? 1
            let span = max(maxValue - minValue, 1)
            let step = proxy.size.width / CGFloat(max(points.count - 1, 1))

            ZStack(alignment: .bottomLeading) {
                VStack(spacing: 0) {
                    ForEach(0..<4, id: \.self) { _ in
                        Rectangle()
                            .fill(AdvisorTheme.line.opacity(0.7))
                            .frame(height: 1)
                        Spacer()
                    }
                }
                Path { path in
                    guard let first = points.first else { return }
                    path.move(to: CGPoint(x: 0, y: proxy.size.height * (1 - CGFloat((first.equity - minValue) / span))))
                    for (index, point) in points.enumerated().dropFirst() {
                        path.addLine(to: CGPoint(
                            x: CGFloat(index) * step,
                            y: proxy.size.height * (1 - CGFloat((point.equity - minValue) / span)),
                        ))
                    }
                }
                .stroke(AdvisorTheme.loss, style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
            }
        }
        .accessibilityLabel("Equity trend chart")
        .accessibilityValue("\(points.last?.equity.advisorCurrency ?? "No data") latest")
    }
}

private struct CardLabel: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 12, weight: .semibold, design: .monospaced))
            .tracking(1.1)
            .foregroundStyle(AdvisorTheme.muted)
    }
}

private struct SectionHeader: View {
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            CardLabel(text: title)
            Spacer()
            Text(detail)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(AdvisorTheme.muted)
        }
    }
}

private struct EmptyCard: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 13, design: .rounded))
            .foregroundStyle(AdvisorTheme.muted)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .advisorCard(radius: 16)
    }
}

private struct QuickActionButton: View {
    let title: String
    let symbol: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: symbol)
                    .font(.headline)
                Text(title)
                    .font(.system(size: 11, weight: .medium, design: .rounded))
            }
            .foregroundStyle(AdvisorTheme.ink)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 13)
            .background(AdvisorTheme.panel)
            .clipShape(RoundedRectangle(cornerRadius: 15, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 15, style: .continuous)
                    .stroke(AdvisorTheme.line, lineWidth: 1)
            }
        }
        .buttonStyle(.plain)
    }
}

struct InlineStatusBanner: View {
    enum Tone { case warning, info }
    let text: String
    let tone: Tone

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: tone == .warning ? "exclamationmark.triangle" : "info.circle")
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
        }
        .font(.system(size: 12, design: .rounded))
        .foregroundStyle(tone == .warning ? AdvisorTheme.loss : AdvisorTheme.ink)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((tone == .warning ? AdvisorTheme.loss : AdvisorTheme.ink).opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 13, style: .continuous))
    }
}

enum AdvisorTheme {
    static let paper = Color(red: 0.969, green: 0.961, blue: 0.941)
    static let panel = Color.white.opacity(0.82)
    static let sunk = Color(red: 0.91, green: 0.90, blue: 0.87)
    static let ink = Color(red: 0.055, green: 0.055, blue: 0.05)
    static let muted = Color(red: 0.39, green: 0.40, blue: 0.37)
    static let line = Color(red: 0.82, green: 0.82, blue: 0.78)
    static let gain = Color(red: 0.02, green: 0.52, blue: 0.38)
    static let loss = Color(red: 0.78, green: 0.16, blue: 0.11)
}

extension View {
    func advisorCard(radius: CGFloat = 18) -> some View {
        self
            .background(AdvisorTheme.panel)
            .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .stroke(AdvisorTheme.line, lineWidth: 1)
            }
            .shadow(color: .black.opacity(0.045), radius: 12, y: 5)
    }
}

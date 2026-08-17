import Charts
import SwiftUI

struct InsightsView: View {
    @EnvironmentObject private var store: AppStore
    @State private var section: InsightSection = .performance

    enum InsightSection: String, CaseIterable, Identifiable {
        case performance = "Performance"
        case breakdowns = "Breakdowns"
        case funnel = "Funnel"
        var id: String { rawValue }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Picker("Insight view", selection: $section) {
                        ForEach(InsightSection.allCases) { item in
                            Text(item.rawValue).tag(item)
                        }
                    }
                    .pickerStyle(.segmented)
                    .accessibilityLabel("Insight view")

                    switch section {
                    case .performance:
                        performance
                    case .breakdowns:
                        breakdowns
                    case .funnel:
                        funnel
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 18)
            }
            .background(AdvisorTheme.paper.ignoresSafeArea())
            .navigationTitle("Insights")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await store.refresh() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Refresh insights")
                }
            }
            .refreshable { await store.refresh() }
        }
    }

    private var performance: some View {
        VStack(alignment: .leading, spacing: 18) {
            summaryCards
            InsightPanel(title: "Equity curve", subtitle: "Paper account snapshots") {
                Chart(store.snapshot.equityHistory) { point in
                    LineMark(
                        x: .value("Date", point.capturedAt),
                        y: .value("Equity", point.equity),
                    )
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(AdvisorTheme.loss)
                    AreaMark(
                        x: .value("Date", point.capturedAt),
                        y: .value("Equity", point.equity),
                    )
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(AdvisorTheme.loss.opacity(0.10))
                }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [3, 4]))
                            .foregroundStyle(AdvisorTheme.line)
                        AxisValueLabel {
                            if let equity = value.as(Double.self) {
                                Text(equity.advisorCurrency)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(AdvisorTheme.muted)
                            }
                        }
                    }
                }
                .frame(height: 220)
                .accessibilityLabel("Equity curve")
                .accessibilityValue("Latest \(store.snapshot.account.equity.advisorCurrency)")
            }

            InsightPanel(title: "Daily P&L", subtitle: "Recent paper sessions") {
                Chart(Array(store.snapshot.equityHistory.suffix(8))) { point in
                    BarMark(
                        x: .value("Date", point.capturedAt, unit: .day),
                        y: .value("P&L", point.dailyPnl ?? 0),
                    )
                    .foregroundStyle((point.dailyPnl ?? 0) >= 0 ? AdvisorTheme.gain : AdvisorTheme.loss)
                }
                .chartXAxis(.hidden)
                .frame(height: 160)
                .accessibilityLabel("Daily P and L chart")
            }
        }
    }

    private var summaryCards: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            InsightStat(label: "Win rate", value: store.snapshot.performance.winRate.advisorPercent)
            InsightStat(label: "Total P&L", value: store.snapshot.performance.totalPnl.advisorSignedCurrency)
            InsightStat(label: "Trades", value: "\(store.snapshot.performance.totalTrades)")
            InsightStat(label: "Max drawdown", value: store.snapshot.performance.maxDrawdown.advisorSignedCurrency)
        }
    }

    private var breakdowns: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Which parts of the strategy are carrying the paper account?")
                .font(.system(size: 15, design: .rounded))
                .foregroundStyle(AdvisorTheme.muted)
            ForEach(store.snapshot.breakdowns) { row in
                VStack(alignment: .leading, spacing: 9) {
                    HStack {
                        Text(row.id)
                            .font(.system(size: 16, weight: .bold, design: .monospaced))
                        Spacer()
                        Text(row.pnl.advisorSignedCurrency)
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .foregroundStyle((row.pnl ?? 0) >= 0 ? AdvisorTheme.gain : AdvisorTheme.loss)
                    }
                    HStack(spacing: 14) {
                        Text("\(row.trades) trades")
                        Text("\(row.winRate.advisorPercent) win rate")
                        if let rr = row.averageRiskReward {
                            Text("\(rr.formatted(.number.precision(.fractionLength(2)))) RR")
                        }
                    }
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(AdvisorTheme.muted)
                }
                .padding(16)
                .advisorCard(radius: 16)
                .accessibilityElement(children: .combine)
                .accessibilityLabel("\(row.id) breakdown")
            }
        }
    }

    private var funnel: some View {
        VStack(alignment: .leading, spacing: 14) {
            InsightPanel(title: "Execution funnel", subtitle: "How signals become paper orders") {
                FunnelStep(label: "Signals detected", value: store.snapshot.funnel.signals, tint: AdvisorTheme.muted, maximum: store.snapshot.funnel.signals)
                FunnelStep(label: "LLM approved", value: store.snapshot.funnel.approved, tint: AdvisorTheme.ink, maximum: store.snapshot.funnel.signals)
                FunnelStep(label: "Orders executed", value: store.snapshot.funnel.executed, tint: AdvisorTheme.gain, maximum: store.snapshot.funnel.signals)
            }
            InsightPanel(title: "Why signals stop", subtitle: "Top validation and execution reasons") {
                ForEach(store.snapshot.funnel.rejectionReasons.sorted { $0.value > $1.value }, id: \.key) { reason, count in
                    HStack {
                        Text(reason)
                            .font(.system(size: 13, design: .rounded))
                            .foregroundStyle(AdvisorTheme.ink)
                        Spacer()
                        Text("\(count)")
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                            .foregroundStyle(AdvisorTheme.muted)
                    }
                }
            }
        }
    }
}

private struct InsightPanel<Content: View>: View {
    let title: String
    let subtitle: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundStyle(AdvisorTheme.ink)
                Text(subtitle)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(AdvisorTheme.muted)
            }
            content()
        }
        .padding(16)
        .advisorCard(radius: 18)
    }
}

private struct InsightStat: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label.uppercased())
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .tracking(0.7)
                .foregroundStyle(AdvisorTheme.muted)
            Text(value)
                .font(.system(size: 19, weight: .semibold, design: .monospaced))
                .foregroundStyle(AdvisorTheme.ink)
                .minimumScaleFactor(0.75)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(15)
        .advisorCard(radius: 16)
    }
}

private struct FunnelStep: View {
    let label: String
    let value: Int
    let tint: Color
    let maximum: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(label)
                    .font(.system(size: 13, design: .rounded))
                Spacer()
                Text("\(value)")
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
            }
            GeometryReader { proxy in
                Capsule()
                    .fill(AdvisorTheme.sunk)
                    .overlay(alignment: .leading) {
                        Capsule()
                            .fill(tint)
                            .frame(width: proxy.size.width * CGFloat(value) / CGFloat(max(maximum, 1)))
                    }
            }
            .frame(height: 7)
        }
    }
}

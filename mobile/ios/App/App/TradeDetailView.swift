import SwiftUI

struct TradeDetailView: View {
    let trade: TradeRecord

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 18) {
                summary
                metrics
                if let reasoning = trade.reasoning {
                    DetailSection(title: "Validation reasoning") {
                        Text(reasoning)
                            .font(.system(size: 15, design: .rounded))
                            .foregroundStyle(AdvisorTheme.ink)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                if let risk = trade.riskAssessment {
                    DetailSection(title: "Risk assessment") {
                        Text(risk)
                            .font(.system(size: 14, design: .rounded))
                            .foregroundStyle(AdvisorTheme.muted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                if !trade.gateResults.isEmpty {
                    DetailSection(title: "Execution gates") {
                        ForEach(trade.gateResults, id: \.self) { gate in
                            Label(gate, systemImage: gate.localizedCaseInsensitiveContains("fail") ? "xmark.circle" : "checkmark.circle")
                                .foregroundStyle(gate.localizedCaseInsensitiveContains("fail") ? AdvisorTheme.loss : AdvisorTheme.gain)
                        }
                    }
                }
                if !trade.vetoFlags.isEmpty {
                    DetailSection(title: "Veto flags") {
                        ForEach(trade.vetoFlags, id: \.self) { flag in
                            Label(flag, systemImage: "exclamationmark.triangle")
                                .foregroundStyle(AdvisorTheme.loss)
                        }
                    }
                }
                fills
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 18)
        }
        .background(AdvisorTheme.paper.ignoresSafeArea())
        .navigationTitle(trade.symbol)
        .navigationBarTitleDisplayMode(.inline)
    }

    private var summary: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(trade.status.uppercased())
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .tracking(1.1)
                        .foregroundStyle(trade.status == "Rejected" ? AdvisorTheme.loss : AdvisorTheme.gain)
                    Text(trade.optionSymbol ?? trade.symbol)
                        .font(.system(size: 18, weight: .bold, design: .monospaced))
                        .foregroundStyle(AdvisorTheme.ink)
                }
                Spacer()
                Text(trade.pnl.advisorSignedCurrency)
                    .font(.system(size: 22, weight: .semibold, design: .monospaced))
                    .foregroundStyle((trade.pnl ?? 0) >= 0 ? AdvisorTheme.gain : AdvisorTheme.loss)
            }
            Text([trade.setup, trade.side, trade.bias].compactMap { $0 }.joined(separator: " · "))
                .font(.system(size: 13, design: .monospaced))
                .foregroundStyle(AdvisorTheme.muted)
            if let reason = trade.exitReason {
                Label(reason, systemImage: trade.status == "Rejected" ? "hand.raised" : "flag")
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(AdvisorTheme.muted)
            }
        }
        .padding(18)
        .advisorCard(radius: 20)
    }

    private var metrics: some View {
        DetailSection(title: "Trade facts") {
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], alignment: .leading, spacing: 16) {
                Metric(label: "Quantity", value: trade.quantity.map { String(Int($0)) } ?? "—")
                Metric(label: "DTE", value: trade.dte.map(String.init) ?? "—")
                Metric(label: "Entry", value: trade.entryPrice?.advisorCurrency ?? "—")
                Metric(label: "Exit", value: trade.exitPrice?.advisorCurrency ?? "—")
                Metric(label: "Return", value: trade.returnPercent?.advisorSignedPercent ?? "—")
                Metric(label: "Decision", value: trade.validationVerdict ?? "—")
            }
        }
    }

    private var fills: some View {
        DetailSection(title: "Fills") {
            if trade.fills.isEmpty {
                Text("No order was submitted.")
                    .font(.system(size: 13, design: .rounded))
                    .foregroundStyle(AdvisorTheme.muted)
            } else {
                ForEach(trade.fills) { fill in
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: fill.kind.localizedCaseInsensitiveContains("exit") ? "arrow.down.right.circle.fill" : "arrow.up.right.circle.fill")
                            .foregroundStyle(fill.pnl.map { $0 >= 0 ? AdvisorTheme.gain : AdvisorTheme.loss } ?? AdvisorTheme.ink)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(fill.kind)
                                .font(.system(size: 13, weight: .semibold, design: .rounded))
                            Text("\(fill.quantity.map { String(Int($0)) } ?? "—") @ \(fill.price?.advisorCurrency ?? "—")")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundStyle(AdvisorTheme.muted)
                            Text(advisorDateTime(fill.timestamp))
                                .font(.system(size: 11, design: .rounded))
                                .foregroundStyle(AdvisorTheme.muted)
                        }
                        Spacer()
                        if let pnl = fill.pnl {
                            Text(pnl.advisorSignedCurrency)
                                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                                .foregroundStyle(pnl >= 0 ? AdvisorTheme.gain : AdvisorTheme.loss)
                        }
                    }
                    .padding(.vertical, 3)
                }
            }
        }
    }
}

private struct DetailSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title.uppercased())
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .tracking(1.1)
                .foregroundStyle(AdvisorTheme.muted)
            content()
        }
        .padding(16)
        .advisorCard(radius: 17)
    }
}

private struct Metric: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(AdvisorTheme.muted)
            Text(value)
                .font(.system(size: 15, weight: .semibold, design: .monospaced))
                .foregroundStyle(AdvisorTheme.ink)
        }
    }
}

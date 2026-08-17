import SwiftUI

struct TradesView: View {
    @EnvironmentObject private var store: AppStore
    @State private var path: [String] = []
    @State private var searchText = ""
    @State private var filter = TradeFilter()
    @State private var showingFilters = false

    private var filteredTrades: [TradeRecord] {
        store.trades.filter { trade in
            let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let matchesQuery = query.isEmpty || [trade.symbol, trade.underlying ?? "", trade.setup ?? "", trade.status]
                .joined(separator: " ")
                .lowercased()
                .contains(query)
            let matchesStatus = filter.status == "All" || trade.status == filter.status
            let matchesSetup = filter.setup == "All" || trade.setup == filter.setup
            let matchesSymbol = filter.symbol == "All" || trade.symbol == filter.symbol
            return matchesQuery && matchesStatus && matchesSetup && matchesSymbol
        }
    }

    var body: some View {
        NavigationStack(path: $path) {
            List {
                if filteredTrades.isEmpty && store.snapshot.brokerOrders.isEmpty {
                    ContentUnavailableView("No matching trades", systemImage: "line.3.horizontal.decrease.circle", description: Text("Try clearing a filter or search term."))
                } else {
                    if !filteredTrades.isEmpty {
                        Section {
                            ForEach(filteredTrades) { trade in
                                NavigationLink(value: trade.id) {
                                    TradeRow(trade: trade)
                                }
                                .accessibilityLabel("\(trade.symbol) \(trade.status) trade")
                                .accessibilityValue(trade.pnl.advisorSignedCurrency)
                            }
                        } header: {
                            HStack {
                                Text("\(filteredTrades.count) records")
                                Spacer()
                                Text(store.networkState.rawValue)
                            }
                            .font(.system(size: 11, design: .monospaced))
                            .textCase(nil)
                        }
                    }

                    if !store.snapshot.brokerOrders.isEmpty {
                        Section("Alpaca paper activity") {
                            ForEach(store.snapshot.brokerOrders) { order in
                                PaperOrderRow(order: order)
                            }
                        }
                    }
                }
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
            .background(AdvisorTheme.paper)
            .navigationTitle("Trades")
            .searchable(text: $searchText, prompt: "Symbol, setup, or status")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showingFilters = true
                    } label: {
                        Image(systemName: filter.isActive ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
                    }
                    .accessibilityLabel("Filter trades")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await store.refresh() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Refresh trades")
                }
            }
            .navigationDestination(for: String.self) { id in
                if let trade = store.trades.first(where: { $0.id == id }) {
                    TradeDetailView(trade: trade)
                } else {
                    ContentUnavailableView("Trade unavailable", systemImage: "questionmark.folder")
                }
            }
            .refreshable { await store.refresh() }
            .sheet(isPresented: $showingFilters) {
                TradeFilterView(filter: $filter, trades: store.trades)
                    .presentationDetents([.medium])
            }
            .onChange(of: store.pendingTradeID) { _, id in
                guard let id else { return }
                if store.selectedTab == .trades {
                    path = [id]
                    store.pendingTradeID = nil
                }
            }
            .onAppear {
                if let id = store.pendingTradeID {
                    path = [id]
                    store.pendingTradeID = nil
                }
            }
        }
    }
}

private struct PaperOrderRow: View {
    let order: PaperOrder

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 7) {
                    Text(order.symbol)
                        .font(.system(size: 15, weight: .bold, design: .monospaced))
                    Text(order.status.uppercased())
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundStyle(order.status.lowercased().contains("reject") ? AdvisorTheme.loss : AdvisorTheme.muted)
                }
                Text([
                    order.side.uppercased(),
                    order.type,
                    order.filledQuantity.map { "\($0.formatted(.number.precision(.fractionLength(0...2)))) filled" },
                ].compactMap { $0 }.joined(separator: " · "))
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(AdvisorTheme.muted)
                Text(advisorDateTime(order.filledAt ?? order.submittedAt))
                    .font(.system(size: 11, design: .rounded))
                    .foregroundStyle(AdvisorTheme.muted)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 5) {
                Text(order.filledAveragePrice.advisorCurrency)
                    .font(.system(size: 14, weight: .semibold, design: .monospaced))
                Text("broker fill")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(AdvisorTheme.muted)
            }
        }
        .padding(.vertical, 5)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(order.symbol) \(order.status) broker order")
        .accessibilityValue("\(order.side) \(order.filledQuantity ?? order.quantity ?? 0) shares or contracts")
    }
}

private struct TradeRow: View {
    let trade: TradeRecord

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 7) {
                    Text(trade.symbol)
                        .font(.system(size: 16, weight: .bold, design: .monospaced))
                    Text(trade.status.uppercased())
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .tracking(0.7)
                        .foregroundStyle(trade.status == "Rejected" ? AdvisorTheme.loss : AdvisorTheme.muted)
                }
                Text([trade.setup, trade.side, trade.dte.map { "\($0) DTE" }].compactMap { $0 }.joined(separator: " · "))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(AdvisorTheme.muted)
                Text(advisorDateTime(trade.entryAt))
                    .font(.system(size: 11, design: .rounded))
                    .foregroundStyle(AdvisorTheme.muted)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 5) {
                Text(trade.pnl.advisorSignedCurrency)
                    .font(.system(size: 15, weight: .semibold, design: .monospaced))
                    .foregroundStyle((trade.pnl ?? 0) >= 0 ? AdvisorTheme.gain : AdvisorTheme.loss)
                Text(trade.validationVerdict ?? "No verdict")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(AdvisorTheme.muted)
            }
        }
        .padding(.vertical, 5)
    }
}

private struct TradeFilterView: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var filter: TradeFilter
    let trades: [TradeRecord]

    private var symbols: [String] { ["All"] + Array(Set(trades.map(\.symbol))).sorted() }
    private var setups: [String] { ["All"] + Array(Set(trades.compactMap(\.setup))).sorted() }
    private let statuses = ["All", "Open", "Closed", "Rejected"]

    var body: some View {
        NavigationStack {
            Form {
                Picker("Status", selection: $filter.status) {
                    ForEach(statuses, id: \.self, content: Text.init)
                }
                Picker("Setup", selection: $filter.setup) {
                    ForEach(setups, id: \.self, content: Text.init)
                }
                Picker("Symbol", selection: $filter.symbol) {
                    ForEach(symbols, id: \.self, content: Text.init)
                }
                Button("Clear filters", role: .destructive) {
                    filter = TradeFilter()
                }
            }
            .navigationTitle("Trade filters")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

import Foundation

enum AppTab: String, CaseIterable, Identifiable {
    case today
    case trades
    case insights
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .today: return "Today"
        case .trades: return "Trades"
        case .insights: return "Insights"
        case .settings: return "Settings"
        }
    }

    var symbol: String {
        switch self {
        case .today: return "sun.max.fill"
        case .trades: return "arrow.left.arrow.right"
        case .insights: return "chart.xyaxis.line"
        case .settings: return "gearshape"
        }
    }
}

enum AppRoute: Hashable {
    case today
    case trades
    case trade(String)
    case insights
    case settings
}

struct AccountSummary: Hashable, Codable {
    var equity: Double?
    var dailyPnl: Double?
    var dailyPnlPercent: Double?
    var buyingPower: Double?
}

struct HealthSummary: Hashable, Codable {
    var isHealthy: Bool
    var heartbeat: Date?
    var loopCount: Int?
    var symbolsTracked: Int?
    var message: String
}

struct EquityPoint: Identifiable, Hashable, Codable {
    var id: String { capturedAt.ISO8601Format() }
    var capturedAt: Date
    var equity: Double
    var dailyPnl: Double?
}

struct PaperPosition: Identifiable, Hashable, Codable {
    let id: String
    var symbol: String
    var optionSymbol: String?
    var side: String
    var quantity: Double
    var entryPrice: Double?
    var currentPrice: Double?
    var unrealizedPnl: Double
    var unrealizedPnlPercent: Double
    var setup: String?
    var dte: Int?
    var openedAt: Date?
    var stopMark: Double?
    var targetMark: Double?
}

struct DecisionRecord: Identifiable, Hashable, Codable {
    let id: String
    var symbol: String
    var setup: String?
    var verdict: String
    var confidence: Double?
    var reasoning: String?
    var createdAt: Date?
}

struct TradeFill: Identifiable, Hashable, Codable {
    let id: String
    var kind: String
    var timestamp: Date?
    var quantity: Double?
    var price: Double?
    var pnl: Double?
    var reason: String?
}

struct TradeRecord: Identifiable, Hashable, Codable {
    let id: String
    var symbol: String
    var underlying: String?
    var optionSymbol: String?
    var side: String?
    var setup: String?
    var status: String
    var dte: Int?
    var quantity: Double?
    var entryPrice: Double?
    var exitPrice: Double?
    var pnl: Double?
    var returnPercent: Double?
    var entryAt: Date?
    var exitAt: Date?
    var exitReason: String?
    var bias: String?
    var validationVerdict: String?
    var confidence: Double?
    var reasoning: String?
    var riskAssessment: String?
    var vetoFlags: [String]
    var gateResults: [String]
    var fills: [TradeFill]
}

struct PerformanceSummary: Hashable, Codable {
    var totalTrades: Int
    var winningTrades: Int
    var losingTrades: Int
    var winRate: Double?
    var totalPnl: Double?
    var maxDrawdown: Double?
    var averageWin: Double?
    var averageLoss: Double?
}

struct BreakdownRow: Identifiable, Hashable, Codable {
    let id: String
    var trades: Int
    var winRate: Double?
    var pnl: Double?
    var averageRiskReward: Double?
}

struct FunnelSummary: Hashable, Codable {
    var signals: Int
    var approved: Int
    var executed: Int
    var approvalRate: Double?
    var rejectionReasons: [String: Int]
}

struct PaperOrder: Identifiable, Hashable, Codable {
    let id: String
    var symbol: String
    var side: String
    var type: String
    var quantity: Double?
    var filledQuantity: Double?
    var limitPrice: Double?
    var stopPrice: Double?
    var filledAveragePrice: Double?
    var status: String
    var submittedAt: Date?
    var filledAt: Date?
}

struct DashboardSnapshot: Hashable, Codable {
    var account: AccountSummary
    var health: HealthSummary
    var equityHistory: [EquityPoint]
    var positions: [PaperPosition]
    var latestDecision: DecisionRecord?
    var performance: PerformanceSummary
    var breakdowns: [BreakdownRow]
    var funnel: FunnelSummary
    var brokerOrders: [PaperOrder]
    var generatedAt: Date
}

struct TradeFilter: Equatable {
    var status: String = "All"
    var setup: String = "All"
    var symbol: String = "All"

    var isActive: Bool {
        status != "All" || setup != "All" || symbol != "All"
    }
}

enum SessionProvider: String {
    case demo = "Demo account"
    case apple = "Apple ID"
    case paper = "Private paper account"
}

enum AppNetworkState: String {
    case live = "Live API"
    case demo = "Demo data"
    case offline = "Offline"
}

extension Double {
    var advisorCurrency: String {
        formatted(.currency(code: "USD").precision(.fractionLength(2)))
    }

    var advisorSignedCurrency: String {
        let value = formatted(.currency(code: "USD").precision(.fractionLength(2)))
        return self > 0 ? "+\(value)" : value
    }

    var advisorPercent: String {
        formatted(.percent.precision(.fractionLength(1)))
    }

    var advisorSignedPercent: String {
        let value = abs(self).formatted(.percent.precision(.fractionLength(1)))
        return self > 0 ? "+\(value)" : self < 0 ? "-\(value)" : value
    }
}

extension Optional where Wrapped == Double {
    var advisorCurrency: String {
        self?.advisorCurrency ?? "—"
    }

    var advisorSignedCurrency: String {
        self?.advisorSignedCurrency ?? "—"
    }

    var advisorPercent: String {
        self?.advisorPercent ?? "—"
    }

    var advisorSignedPercent: String {
        self?.advisorSignedPercent ?? "—"
    }
}

func advisorRelativeTime(_ date: Date?) -> String {
    guard let date else { return "No reading" }
    return date.formatted(.relative(presentation: .named))
}

func advisorDateTime(_ date: Date?) -> String {
    guard let date else { return "—" }
    return date.formatted(date: .abbreviated, time: .shortened)
}

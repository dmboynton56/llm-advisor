import Foundation

/// The native screens run on local fixtures in demo mode, but production data
/// flows through this small authenticated boundary rather than embedding any
/// Supabase or broker credentials in the app.
struct MobileAPIClient {
    struct DemoPairResponse: Decodable {
        let accessToken: String
        let expiresAt: Date
        let provider: String
        let environment: String
        let readOnly: Bool
    }

    struct AppleSessionResponse: Decodable {
        let accessToken: String
        let refreshToken: String?
        let expiresIn: Int?
    }

    struct BootstrapResponse: Decodable {
        let schemaVersion: Int
        let generatedAt: Date
        let account: AccountSummary
        let health: HealthSummary
        let equityHistory: [EquityPoint]
        let positions: [PaperPosition]
        let latestDecision: DecisionRecord?
        let performance: PerformanceSummary
        let breakdowns: [BreakdownRow]
        let funnel: FunnelSummary
        let trades: [TradeRecord]
    }

    struct AlpacaPaperResponse: Decodable {
        struct Account: Decodable {
            let equity: Double?
            let lastEquity: Double?
            let buyingPower: Double?
            let dailyPnl: Double?
            let dailyPnlPercent: Double?

            enum CodingKeys: String, CodingKey {
                case equity
                case lastEquity = "last_equity"
                case buyingPower = "buying_power"
                case dailyPnl = "daily_pnl"
                case dailyPnlPercent = "daily_pnl_pct"
            }
        }

        struct Position: Decodable {
            let symbol: String
            let optionSymbol: String?
            let underlyingSymbol: String?
            let side: String
            let quantity: Double
            let entryPrice: Double?
            let currentPrice: Double?
            let unrealizedPnl: Double
            let unrealizedPnlPercent: Double
            let setup: String?
            let dte: Int?
            let openedAt: Date?
            let stopMark: Double?
            let targetMark: Double?

            enum CodingKeys: String, CodingKey {
                case symbol
                case optionSymbol = "option_symbol"
                case underlyingSymbol = "underlying_symbol"
                case side
                case quantity = "qty"
                case entryPrice = "entry_price"
                case currentPrice = "current_price"
                case unrealizedPnl = "unrealized_pl"
                case unrealizedPnlPercent = "unrealized_plpc"
                case setup = "setup_type"
                case dte
                case openedAt = "opened_at"
                case stopMark = "stop_mark"
                case targetMark = "tp_mark"
            }
        }

        struct Order: Decodable {
            let id: String?
            let symbol: String
            let side: String
            let type: String
            let quantity: Double?
            let filledQuantity: Double?
            let limitPrice: Double?
            let stopPrice: Double?
            let filledAveragePrice: Double?
            let status: String
            let submittedAt: Date?
            let filledAt: Date?

            enum CodingKeys: String, CodingKey {
                case id
                case symbol
                case side
                case type
                case quantity = "qty"
                case filledQuantity = "filled_qty"
                case limitPrice = "limit_price"
                case stopPrice = "stop_price"
                case filledAveragePrice = "filled_avg_price"
                case status
                case submittedAt = "submitted_at"
                case filledAt = "filled_at"
            }
        }

        let schemaVersion: Int
        let provider: String
        let environment: String
        let readOnly: Bool
        let fetchedAt: Date
        let account: Account
        let positions: [Position]
        let openOrders: [Order]
        let recentOrders: [Order]

        enum CodingKeys: String, CodingKey {
            case schemaVersion
            case provider
            case environment
            case readOnly
            case fetchedAt
            case account
            case positions
            case openOrders
            case recentOrders
        }
    }

    enum ClientError: LocalizedError, Equatable {
        case invalidResponse
        case unauthorized

        var errorDescription: String? {
            switch self {
            case .invalidResponse: return "The mobile API returned an invalid response."
            case .unauthorized: return "The mobile session is no longer authorized."
            }
        }
    }

    var baseURL: URL
    var session: URLSession = .shared

    init(baseURL: URL? = nil) {
        #if DEBUG
        if let index = ProcessInfo.processInfo.arguments.firstIndex(of: "-mobile-api-base-url"),
           ProcessInfo.processInfo.arguments.indices.contains(index + 1),
           let override = URL(string: ProcessInfo.processInfo.arguments[index + 1]) {
            self.baseURL = override
            return
        }
        #endif
        self.baseURL = baseURL ?? URL(string: "https://llm-advisor.drewboynton.com")!
    }

    func fetchBootstrap(accessToken: String) async throws -> BootstrapResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/mobile/v1/bootstrap"))
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ClientError.invalidResponse }
        guard http.statusCode != 401 else { throw ClientError.unauthorized }
        guard (200..<300).contains(http.statusCode) else { throw ClientError.invalidResponse }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(BootstrapResponse.self, from: data)
    }

    func pairPrivatePaperAccount(code: String) async throws -> DemoPairResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/mobile/v1/dev/pair"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["code": code])
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ClientError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw ClientError.unauthorized }
            throw ClientError.invalidResponse
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(DemoPairResponse.self, from: data)
    }

    func fetchAlpacaPaper(accessToken: String) async throws -> AlpacaPaperResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/mobile/v1/alpaca"))
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ClientError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw ClientError.unauthorized }
            throw ClientError.invalidResponse
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(AlpacaPaperResponse.self, from: data)
    }

    func exchangeAppleIdentityToken(_ identityToken: String, nonce: String?) async throws -> AppleSessionResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/mobile/v1/auth/apple"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        var payload: [String: String] = ["identityToken": identityToken]
        if let nonce { payload["nonce"] = nonce }
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ClientError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw ClientError.unauthorized }
            throw ClientError.invalidResponse
        }
        let decoder = JSONDecoder()
        return try decoder.decode(AppleSessionResponse.self, from: data)
    }
}

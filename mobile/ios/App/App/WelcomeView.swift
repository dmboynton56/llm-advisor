import AuthenticationServices
import CryptoKit
import SwiftUI

struct WelcomeView: View {
    @EnvironmentObject private var store: AppStore
    @State private var signInError: String?
    @State private var appleNonce: String?
    @State private var isSigningIn = false
    @State private var showingPaperPairing = false

    var body: some View {
        ZStack {
            AdvisorTheme.paper.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    Spacer(minLength: 44)
                    Image("LLMAdvisorMark")
                        .resizable()
                        .scaledToFill()
                        .frame(width: 68, height: 68)
                        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                    VStack(alignment: .leading, spacing: 8) {
                        Text("LLM ADVISOR")
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                            .tracking(1.5)
                            .foregroundStyle(AdvisorTheme.muted)
                        Text("Your paper-trading\nbriefing, native on iPhone.")
                            .font(.system(size: 34, weight: .bold, design: .rounded))
                            .foregroundStyle(AdvisorTheme.ink)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Text("Glance at loop health, equity, positions, decisions, and evidence without carrying the whole operations dashboard in your pocket.")
                        .font(.system(size: 17, design: .rounded))
                        .foregroundStyle(AdvisorTheme.muted)
                        .fixedSize(horizontal: false, vertical: true)

                    VStack(spacing: 12) {
                        SignInWithAppleButton(.signIn) { request in
                            request.requestedScopes = [.email, .fullName]
                            let nonce = UUID().uuidString
                            appleNonce = nonce
                            request.nonce = Self.sha256(nonce)
                        } onCompletion: { result in
                            switch result {
                            case .success(let authorization):
                                guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential,
                                      let tokenData = credential.identityToken,
                                      let token = String(data: tokenData, encoding: .utf8) else {
                                    signInError = "Apple did not return an identity token."
                                    return
                                }
                                isSigningIn = true
                                Task {
                                    await store.completeAppleSignIn(identityToken: token, nonce: appleNonce)
                                    isSigningIn = false
                                }
                            case .failure(let error):
                                signInError = error.localizedDescription
                            }
                        }
                        .signInWithAppleButtonStyle(.black)
                        .frame(height: 52)
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .disabled(isSigningIn)
                        .accessibilityLabel("Sign in with Apple")

                        Button {
                            store.continueWithDemo()
                        } label: {
                            HStack {
                                Image(systemName: "sparkles")
                                Text("Continue with demo data")
                                    .fontWeight(.semibold)
                                Spacer()
                                Image(systemName: "arrow.right")
                            }
                            .foregroundStyle(AdvisorTheme.paper)
                            .padding(.horizontal, 18)
                            .frame(height: 52)
                            .background(AdvisorTheme.ink)
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                        .buttonStyle(.plain)
                        .accessibilityHint("Uses local demo data so you can explore every screen in the simulator")

                        #if DEBUG
                        Button {
                            showingPaperPairing = true
                        } label: {
                            HStack {
                                Image(systemName: "server.rack")
                                Text("Connect private paper account")
                                    .fontWeight(.semibold)
                                Spacer()
                                Image(systemName: "lock.shield")
                            }
                            .foregroundStyle(AdvisorTheme.ink)
                            .padding(.horizontal, 18)
                            .frame(height: 52)
                            .background(AdvisorTheme.panel)
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                            .overlay {
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .stroke(AdvisorTheme.line, lineWidth: 1)
                            }
                        }
                        .buttonStyle(.plain)
                        .accessibilityHint("Uses a short-lived development pairing code to read the server-owned Alpaca paper account")
                        #endif
                    }

                    if let signInError {
                        InlineStatusBanner(text: signInError, tone: .warning)
                    }
                    if let error = store.lastError, !store.isAuthenticated {
                        InlineStatusBanner(text: error, tone: .warning)
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Label("Read-only by design", systemImage: "checkmark.shield")
                        Label("No broker credentials in the app", systemImage: "lock.shield")
                        Label("Notifications and Live Activities are optional", systemImage: "bell.badge")
                    }
                    .font(.system(size: 13, design: .rounded))
                    .foregroundStyle(AdvisorTheme.muted)
                    .padding(.top, 8)

                    Text("Apple sign-in is wired for the production path. Demo mode is available while the private API and Apple Developer capabilities are being configured.")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(AdvisorTheme.muted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 36)
            }
        }
        .sheet(isPresented: $showingPaperPairing) {
            PaperPairingView()
                .environmentObject(store)
                .presentationDetents([.medium])
        }
    }

    private static func sha256(_ value: String) -> String {
        SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

#if DEBUG
private struct PaperPairingView: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @State private var code = ""

    private var canSubmit: Bool {
        !code.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !store.isPairing
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text("Enter the short-lived code generated for this private development build. The app receives a read-only paper-account token; Alpaca credentials never leave the server.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    SecureField("Pairing code", text: $code)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textContentType(.oneTimeCode)
                        .accessibilityLabel("Private paper pairing code")
                }

                if let error = store.lastError {
                    Section {
                        InlineStatusBanner(text: error, tone: .warning)
                    }
                }

                Section {
                    Button {
                        Task {
                            await store.pairPrivatePaperAccount(code: code)
                            if store.provider == .paper {
                                dismiss()
                            }
                        }
                    } label: {
                        HStack {
                            Text(store.isPairing ? "Connecting…" : "Connect paper account")
                            Spacer()
                            if store.isPairing {
                                ProgressView()
                            }
                        }
                    }
                    .disabled(!canSubmit)
                }
            }
            .navigationTitle("Private paper access")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
#endif

struct NativeAppView: View {
    @StateObject private var store = AppStore()

    var body: some View {
        Group {
            if store.isAuthenticated {
                TabView(selection: $store.selectedTab) {
                    TodayView()
                        .tabItem { Label(AppTab.today.title, systemImage: AppTab.today.symbol) }
                        .tag(AppTab.today)
                    TradesView()
                        .tabItem { Label(AppTab.trades.title, systemImage: AppTab.trades.symbol) }
                        .tag(AppTab.trades)
                    InsightsView()
                        .tabItem { Label(AppTab.insights.title, systemImage: AppTab.insights.symbol) }
                        .tag(AppTab.insights)
                    SettingsView()
                        .tabItem { Label(AppTab.settings.title, systemImage: AppTab.settings.symbol) }
                        .tag(AppTab.settings)
                }
                .tint(AdvisorTheme.ink)
                .environmentObject(store)
            } else {
                WelcomeView()
                    .environmentObject(store)
            }
        }
        .onOpenURL { url in
            store.handleDeepLink(url)
        }
        .onReceive(NotificationCenter.default.publisher(for: .advisorDeepLink)) { notification in
            if let url = notification.object as? URL {
                store.handleDeepLink(url)
            }
        }
    }
}

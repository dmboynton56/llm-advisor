# iOS native implementation plan

## Product direction

Treat the first iOS build as a **pocket operations companion** for the
paper-trading system: glanceable account health, current positions, recent
decisions, and post-session review. It is an observability surface, not a
trade-entry terminal.

## Phase 0 — native simulator demo (complete)

- Standalone SwiftUI target, iOS 18+, with no Capacitor runtime.
- Native Today, Trades, trade detail, Insights, and Settings surfaces.
- Local demo fixtures so every screen is explorable without credentials.
- Native refresh, offline state, filters, deep links, notifications, and
  ActivityKit session controls.
- Debug-only pairing to the server-owned Alpaca paper account, with a short-lived
  read-only token stored in Keychain and no broker credentials in the app.
- Existing LLM Advisor identity, icon, launch styling, and read-only safety
  boundary.

Success means a developer can clone the repository, run `npm install` in
`mobile`, run `npm run run:simulator`, and exercise the complete native demo.

## Phase 1 — production identity and TestFlight shell (next)

- Finish Apple Developer team configuration and validate the Sign in with Apple
  exchange on a physical device.
- Run `sql/011_mobile_app.sql` and enable the private user allow-list for the
  mobile API.
- Replace the development pairing session with a persisted production
  access/refresh-token session and secure keychain storage.
- Add privacy metadata, support links, versioning, signing, and TestFlight CI.
- Add a WidgetKit extension to render the `AdvisorSessionAttributes` Live
  Activity on Lock Screen and Dynamic Island.
- Refine compact navigation and safe-area behavior on physical devices.

## Phase 2 — native companion experience (API-backed)

- The versioned API boundary is now available at `/api/mobile/v1/bootstrap`,
  `/live`, `/trades`, `/insights`, and `/auth/apple`.
- Hydrate native Today/Trades/Insights from those responses after authentication
  is enabled.
- Add APNs device registration and server-triggered high-signal outbox delivery.
- Keep execution out of scope until authentication, authorization, audit logs,
  and explicit risk controls have been designed and reviewed.

## Architecture decision

The dashboard remains a Next.js application with server-side Supabase and
Alpaca access. The native target owns the user-facing information architecture
and calls a small authenticated mobile API. This lets the web UI be retired
after native parity without moving secrets, broker logic, or server-only
business rules into the app bundle.

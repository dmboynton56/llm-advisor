# Native iOS implementation

The mobile target is now a standalone SwiftUI application (iOS 18+) with no
Capacitor runtime. `SceneDelegate` owns a `UIHostingController` root and
`NativeAppView` owns the tab/navigation shell:

- **Today**: calm briefing, loop health, paper equity, open positions, latest decision, refresh, and alert shortcut.
- **Trades**: searchable/filterable native `List`, drill-down detail, fills, validation gates, reasoning, and veto flags.
- **Insights**: performance, equity curve, breakdowns, and execution funnel.
- **Settings**: Sign in with Apple entry point, demo/offline mode, notification permission and test alerts, Live Activity controls, and sign out.

The simulator uses `DemoData` so every workflow is testable without credentials.
The Debug build also supports a private pairing code that returns a short-lived,
read-only token for the server-owned Alpaca paper account. Production mode is
intentionally read-only: `MobileAPIClient` calls the authenticated mobile API,
while the Next.js service keeps Supabase and broker credentials on the server.

## API boundary

The native client expects an access token in a Bearer header. Production routes
verify it through Supabase Auth and optionally restrict it with
`MOBILE_ALLOWED_USER_IDS` (comma-separated UUIDs); the Debug pairing route uses
an explicitly enabled, signed development token:

| Route | Purpose |
| --- | --- |
| `POST /api/mobile/v1/dev/pair` | Debug-only pairing for the private paper account |
| `GET /api/mobile/v1/bootstrap` | Full Today + Trades + Insights payload |
| `GET /api/mobile/v1/alpaca` | Read-only Alpaca paper account, positions, and orders |
| `GET /api/mobile/v1/live` | Small health/account/positions payload for refresh and future widgets |
| `GET /api/mobile/v1/trades` | Filtered trade history (`status`, `setup`, `symbol`, `limit`) |
| `GET /api/mobile/v1/insights` | Performance, curve, breakdown, and funnel payload |

Run `sql/011_mobile_app.sql` in Supabase before enabling device registration or
server-triggered APNs delivery. The migration intentionally grants those tables
only to `service_role`.

## Notifications and Live Activity

Local high-signal notification flows are live in the simulator. Notification
payloads use `llmadvisor://trades/<id>` and `llmadvisor://health`, which the app
routes to the native destination. `LiveActivityManager` defines a discreet
monitoring session state and intentionally excludes symbols, dollars, and P&L
from Lock Screen content.

Before a physical-device/TestFlight build, configure an Apple Developer team,
Sign in with Apple capability, APNs capability, and a WidgetKit extension that
renders `AdvisorSessionAttributes`. The target compiles for `iphoneos`; a
physical install still requires the Xcode team/provisioning selection and a
trusted device.

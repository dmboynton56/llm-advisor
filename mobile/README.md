# LLM Advisor native iOS app

This is a native SwiftUI read-only companion for the LLM Advisor paper-trading
loop. The app owns navigation, state, notifications, deep links, and the
briefing experience; the existing Next.js application remains a temporary
authenticated API service while native parity is completed.

## Run in the iOS Simulator

Requirements: Node.js, Xcode, and an installed iOS Simulator runtime.

```bash
cd mobile
npm install
npm run build:simulator
open ios/App/App.xcodeproj
```

In Xcode, select an iPhone simulator and press **Run**. The first screen is
the native Today briefing. Demo mode is intentionally fixture-backed so all
workflows are available without credentials; the production path hydrates the
same screens from the authenticated mobile API.

For the shortest path after `npm install`, run the full build, install, and
launch flow with:

```bash
npm run run:simulator
```

To point the Debug simulator at a local Next.js server, start the server in
`web` with the paper-account variables below, then launch with:

    MOBILE_API_BASE_URL=http://127.0.0.1:3000 npm run run:simulator

The override is Debug-only; a release build uses the deployed API URL.

Set `IOS_SIMULATOR_UDID` first if you want a simulator other than the first
available iPhone.

## Run on a physical iPhone

Open `ios/App/App.xcodeproj` in Xcode, connect an unlocked iPhone, choose the
phone as the run destination, and set the **App** target's Signing &
Capabilities team to your Apple Developer team. Press **Run**, then accept the
trust/developer prompts on the phone. The Debug pairing control is available
there as well. For a physical phone, use the deployed HTTPS API URL; a local
`127.0.0.1` URL points back to the phone, not your Mac.

Regenerate the app icon after changing the source brand image:

```bash
./scripts/generate-ios-assets.sh
```

## Current scope

- Native Today, Trades, trade detail, Insights (performance/breakdowns/funnel),
  and Settings screens
- Branded home-screen icon and launch screen
- Development-only private paper pairing with short-lived Keychain-backed
  sessions; Alpaca credentials stay on the server
- Demo fixtures plus an authenticated `/api/mobile/v1/bootstrap` client boundary
- Sign in with Apple UI, local high-signal notifications, deep-link routing, and
  Live Activity scaffolding
- No broker credentials stored in the iOS app
- No live-money execution

## Connect the server-owned paper account

The native Debug build can read the server-owned Alpaca paper account before
Sign in with Apple is configured. The server must explicitly opt into the
development path. Configure these server-only variables:

    MOBILE_DEMO_MODE=true
    MOBILE_DEMO_PAIRING_CODE=use-a-new-short-lived-code
    MOBILE_DEMO_TOKEN_SECRET=use-a-long-random-secret
    MOBILE_DEMO_TOKEN_TTL_SECONDS=43200
    ALPACA_PAPER_TRADING=true

Keep the Alpaca key and secret server-only. After deploying those variables,
open the Debug app, choose Connect private paper account, and enter the
pairing code. The app stores only the short-lived read-only token in the iOS
Keychain. It then reads account equity, buying power, positions, open orders,
and recent paper fills through the mobile API.

For local Next.js development, put the variables in `web/.env.local` (not in
the iOS project). Include the existing `ALPACA_API_KEY` and
`ALPACA_SECRET_KEY` there as well, then restart `npm run dev`. The root
repository `.env` is not automatically loaded by Next.js when the project is
started from `web`; source it explicitly or copy the required values into
`web/.env.local`.

For the deployed app, open the `llm-advisor` project in Vercel, go to
**Settings → Environment Variables**, add the variables above to **Production**,
and redeploy the application. The new `/api/mobile/v1/dev/pair` and
`/api/mobile/v1/alpaca` routes must be present in that deployment before the
iPhone can use them.

Rotate MOBILE_DEMO_PAIRING_CODE after pairing the test devices. This
development route is intentionally not a substitute for production identity;
Sign in with Apple can be added later without changing the Alpaca data path.

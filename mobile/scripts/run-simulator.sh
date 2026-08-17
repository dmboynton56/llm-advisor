#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MOBILE_DIR=$(dirname "$SCRIPT_DIR")
APP_PATH="$MOBILE_DIR/build/Build/Products/Debug-iphonesimulator/App.app"
BUNDLE_ID="com.drewboynton.llmadvisor"
DEVICE_UDID=${IOS_SIMULATOR_UDID:-}

cd "$MOBILE_DIR"

if [ -z "$DEVICE_UDID" ]; then
  DEVICE_UDID=$(xcrun simctl list devices available | awk -F '[()]' '/iPhone/ { print $2; exit }')
fi

if [ -z "$DEVICE_UDID" ]; then
  echo "No available iPhone simulator was found." >&2
  exit 1
fi

DEVICE_STATE=$(xcrun simctl list devices | awk -v id="$DEVICE_UDID" '$0 ~ id { print $(NF); exit }')
if [ "$DEVICE_STATE" != "(Booted)" ]; then
  xcrun simctl boot "$DEVICE_UDID"
fi

xcrun simctl bootstatus "$DEVICE_UDID" -b
npm run build:simulator
xcrun simctl install "$DEVICE_UDID" "$APP_PATH"
if [ -n "${MOBILE_API_BASE_URL:-}" ]; then
  xcrun simctl launch --terminate-running-process "$DEVICE_UDID" "$BUNDLE_ID" \
    -mobile-api-base-url "$MOBILE_API_BASE_URL"
else
  xcrun simctl launch --terminate-running-process "$DEVICE_UDID" "$BUNDLE_ID"
fi

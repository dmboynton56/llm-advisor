#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MOBILE_DIR=$(dirname "$SCRIPT_DIR")
SOURCE_ICON="$MOBILE_DIR/../web/public/llm-advisor-logo.png"
SOURCE_MARK="$MOBILE_DIR/../web/public/llm-advisor-mark.png"
APPICON_DIR="$MOBILE_DIR/ios/App/App/Assets.xcassets/AppIcon.appiconset"
SPLASH_DIR="$MOBILE_DIR/ios/App/App/Assets.xcassets/Splash.imageset"
MARK_DIR="$MOBILE_DIR/ios/App/App/Assets.xcassets/LLMAdvisorMark.imageset"
TEMP_DIR=$(mktemp -d)

trap 'rm -rf "$TEMP_DIR"' EXIT

if [ ! -f "$SOURCE_ICON" ]; then
  echo "Missing source icon: $SOURCE_ICON" >&2
  exit 1
fi

if [ ! -f "$SOURCE_MARK" ]; then
  echo "Missing source mark: $SOURCE_MARK" >&2
  exit 1
fi

if [ ! -d "$APPICON_DIR" ]; then
  echo "Missing iOS app icon catalog. Run 'npx cap add ios' first." >&2
  exit 1
fi

if [ ! -d "$SPLASH_DIR" ]; then
  echo "Missing iOS splash catalog. Run 'npx cap add ios' first." >&2
  exit 1
fi

mkdir -p "$MARK_DIR"

# The native target uses one universal 1024px marketing icon.
# sips keeps asset generation dependency-free and reproducible on macOS.
sips --resampleHeightWidth 1024 1024 "$SOURCE_ICON" \
  --out "$APPICON_DIR/AppIcon-512@2x.png" >/dev/null

# Center a compact brand tile on the app's warm paper background for launch.
sips --resampleHeightWidth 640 640 "$SOURCE_ICON" \
  --out "$TEMP_DIR/splash-logo.png" >/dev/null
sips --padToHeightWidth 2732 2732 --padColor F7F5F0 \
  "$TEMP_DIR/splash-logo.png" --out "$TEMP_DIR/splash.png" >/dev/null

for splash in \
  splash-2732x2732.png \
  splash-2732x2732-1.png \
  splash-2732x2732-2.png
do
  cp "$TEMP_DIR/splash.png" "$SPLASH_DIR/$splash"
done

sips --resampleHeightWidth 512 512 "$SOURCE_MARK" \
  --out "$MARK_DIR/mark.png" >/dev/null

echo "Generated $APPICON_DIR/AppIcon-512@2x.png"
echo "Generated branded launch images in $SPLASH_DIR"
echo "Generated native header mark in $MARK_DIR"

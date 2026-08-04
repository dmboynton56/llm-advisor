/**
 * Recharts renders these straight onto SVG presentation attributes and inline
 * styles, so `var(--token)` resolves at paint time. That means charts re-theme
 * with the rest of the page on a [data-theme] flip — no JS, no re-render.
 */
export const AXIS_TICK = {
  fill: "var(--ink-3)",
  fontSize: 11,
  fontFamily: "var(--font-plex-mono)",
} as const;

export const GRID_STROKE = "var(--line)";
export const AXIS_LINE = "var(--line-2)";

export const TOOLTIP_CONTENT_STYLE = {
  backgroundColor: "var(--card)",
  border: "1px solid var(--line-2)",
  borderRadius: 10,
  boxShadow: "var(--shadow-2)",
  fontSize: 12,
  fontFamily: "var(--font-plex-mono)",
  color: "var(--ink)",
} as const;

export const TOOLTIP_LABEL_STYLE = { color: "var(--ink-3)" } as const;

export const TOOLTIP_CURSOR_FILL = "color-mix(in srgb, var(--ink) 7%, transparent)";

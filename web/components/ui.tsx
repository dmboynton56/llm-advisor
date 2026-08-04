import type { ReactNode } from "react";
import clsx from "clsx";

export type Tone = "neutral" | "positive" | "negative";

export function toneClass(tone: Tone): string {
  if (tone === "positive") return "text-gain";
  if (tone === "negative") return "text-loss";
  return "";
}

/** Signed numbers decide their own tone often enough to be worth a helper. */
export function toneOf(value: number | null | undefined): Tone {
  if (value === null || value === undefined || value === 0) return "neutral";
  return value > 0 ? "positive" : "negative";
}

/** Elevated surface. Light mode separates with shadow, dark mode with a hairline. */
export function Panel({
  children,
  className,
  padded = true,
}: {
  children: ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <div
      className={clsx(
        "rounded-panel-lg border border-panel-border bg-card shadow-panel",
        padded && "p-5",
        className,
      )}
    >
      {children}
    </div>
  );
}

/**
 * A titled region of the page. The title sits on the paper, not inside a box —
 * the box (if any) is whatever the caller renders as children.
 */
export function Section({
  title,
  subtitle,
  figure,
  children,
  className,
}: {
  title: string;
  subtitle?: ReactNode;
  figure?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={clsx("mt-11", className)}>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-[16.5px] font-semibold">{title}</h2>
          {subtitle ? (
            <p className="mt-1 max-w-[62ch] text-[12.5px] text-ink-3">{subtitle}</p>
          ) : null}
        </div>
        {figure ? (
          <span className="num whitespace-nowrap text-[19px] font-medium tracking-[-0.025em]">
            {figure}
          </span>
        ) : null}
      </div>
      {children}
    </section>
  );
}

/** Page heading. */
export function PageHeader({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="mb-8">
      <h1 className="text-[26px] font-semibold tracking-[-0.025em]">{title}</h1>
      {children ? (
        <p className="mt-2.5 max-w-[70ch] text-[13.5px] text-ink-2">{children}</p>
      ) : null}
    </div>
  );
}

/**
 * Hairline-separated figures. No boxes — the rules do the separating, which
 * keeps a row of four numbers from reading as four competing cards.
 */
export function StatRow({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={clsx("stat-row", className)}>{children}</div>;
}

export function Stat({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: Tone;
}) {
  // Padding and rules come from `.stat-row > *`.
  return (
    <div>
      <p className="tag">{label}</p>
      <p
        className={clsx(
          "num mt-2 text-[21px] font-medium tracking-[-0.025em]",
          toneClass(tone),
        )}
      >
        {value}
      </p>
      {hint ? <p className="mt-1.5 text-[11.5px] text-ink-3">{hint}</p> : null}
    </div>
  );
}

/** Figure inside a Panel, for the rail and for grids. */
export function PanelStat({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: Tone;
}) {
  return (
    <div>
      <p className="tag">{label}</p>
      <p
        className={clsx(
          "num mt-2 text-[20px] font-medium tracking-[-0.028em]",
          toneClass(tone),
        )}
      >
        {value}
      </p>
      {hint ? <p className="mt-1.5 text-[11.5px] text-ink-3">{hint}</p> : null}
    </div>
  );
}

export function PanelHead({
  title,
  aside,
}: {
  title: string;
  aside?: ReactNode;
}) {
  return (
    <div className="mb-3.5 flex items-baseline justify-between gap-2.5">
      <h2 className="text-[13px] font-semibold tracking-[-0.008em]">{title}</h2>
      {aside ? <span className="tag">{aside}</span> : null}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex min-h-24 items-center justify-center rounded-panel border border-dashed border-line-2 p-6 text-center text-[13px] text-ink-3">
      {message}
    </div>
  );
}

/** Inline proportion meter, used wherever a rate needs a shape as well as a number. */
export function Meter({ value }: { value: number | null | undefined }) {
  const pct =
    value === null || value === undefined || Number.isNaN(value)
      ? null
      : Math.round(Math.min(Math.max(value, 0), 1) * 100);
  return (
    <span className="inline-flex items-center gap-2.5">
      <span className="h-1 w-[42px] overflow-hidden rounded-full bg-sunk">
        <span
          className="block h-full rounded-full bg-ink-2"
          style={{ width: `${pct ?? 0}%` }}
        />
      </span>
      <span className="num text-[12px] text-ink-2">
        {pct === null ? "—" : `${pct}%`}
      </span>
    </span>
  );
}

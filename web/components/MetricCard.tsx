import type { ReactNode } from "react";
import clsx from "clsx";

export function MetricCard({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "neutral" | "positive" | "negative";
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">
        {label}
      </p>
      <p
        className={clsx(
          "mt-1 text-2xl font-semibold tabular-nums tracking-tight",
          tone === "positive" && "text-emerald-400",
          tone === "negative" && "text-rose-400",
        )}
      >
        {value}
      </p>
      {hint ? <p className="mt-1 text-xs text-zinc-500">{hint}</p> : null}
    </div>
  );
}

export function Card({
  title,
  subtitle,
  children,
  className,
}: {
  title?: string;
  subtitle?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={clsx(
        "rounded-xl border border-zinc-800 bg-zinc-900/50 p-4",
        className,
      )}
    >
      {title ? (
        <div className="mb-3">
          <h2 className="text-sm font-semibold text-zinc-200">{title}</h2>
          {subtitle ? (
            <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>
          ) : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex min-h-24 items-center justify-center rounded-lg border border-dashed border-zinc-800 p-6 text-center text-sm text-zinc-500">
      {message}
    </div>
  );
}

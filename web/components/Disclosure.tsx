"use client";

import { useId, useState, type ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import clsx from "clsx";

export type DisclosureAggregate = {
  label: string;
  value: ReactNode;
  tone?: "neutral" | "positive" | "negative";
};

/**
 * Collapsed detail table. The closed row still carries the headline numbers so
 * collapsing hides the rows, not the answer.
 */
export function Disclosure({
  title,
  subtitle,
  aggregates = [],
  defaultOpen = false,
  children,
}: {
  title: string;
  subtitle?: string;
  aggregates?: DisclosureAggregate[];
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();

  return (
    <div className="overflow-hidden rounded-panel-lg border border-panel-border bg-card shadow-panel">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={bodyId}
        className="flex w-full items-center gap-3.5 p-[18px] text-left transition-colors hover:bg-sunk"
      >
        <ChevronRight
          aria-hidden
          className={clsx(
            "size-[13px] shrink-0 text-ink-3 transition-transform duration-200",
            open && "rotate-90",
          )}
        />
        <span>
          <span className="block text-[14.5px] font-semibold tracking-[-0.012em]">
            {title}
          </span>
          {subtitle ? (
            <span className="mt-0.5 block text-[12px] text-ink-3">{subtitle}</span>
          ) : null}
        </span>

        {aggregates.length > 0 ? (
          <span className="ml-auto hidden items-center gap-6 sm:flex">
            {aggregates.map((agg) => (
              <span key={agg.label} className="text-right">
                <span className="tag block">{agg.label}</span>
                <span
                  className={clsx(
                    "num mt-1 block text-[14.5px] font-medium",
                    agg.tone === "positive" && "text-gain",
                    agg.tone === "negative" && "text-loss",
                  )}
                >
                  {agg.value}
                </span>
              </span>
            ))}
          </span>
        ) : null}
      </button>

      {/* grid-template-rows animates cleanly from an unknown content height. */}
      <div
        id={bodyId}
        className={clsx(
          "grid transition-[grid-template-rows] duration-300 ease-out",
          open ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
        )}
      >
        <div className="overflow-hidden">
          <div className="border-t border-line">{children}</div>
        </div>
      </div>
    </div>
  );
}

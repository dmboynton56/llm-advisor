"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { Activity, BarChart3, Command, Filter, LineChart, Table2 } from "lucide-react";
import clsx from "clsx";

const navItems = [
  { href: "/", label: "Overview", icon: Activity },
  { href: "/trades", label: "Trades", icon: Table2 },
  { href: "/breakdowns", label: "Breakdowns", icon: BarChart3 },
  { href: "/funnel", label: "Funnel", icon: Filter },
  ...(process.env.NEXT_PUBLIC_COMMAND_CENTER_ENABLED === "true"
    ? [{ href: "/command-center", label: "Command Center", icon: Command }]
    : []),
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-zinc-800/80 bg-zinc-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
          <Link href="/" className="flex items-center gap-2">
            <span className="flex size-8 items-center justify-center rounded-lg bg-emerald-500/15 text-emerald-400">
              <LineChart className="size-4" />
            </span>
            <span className="text-sm font-semibold tracking-tight">
              LLM Advisor
              <span className="ml-2 hidden text-zinc-500 sm:inline">
                Ops Dashboard
              </span>
            </span>
          </Link>
          <nav className="flex items-center gap-1 overflow-x-auto">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={clsx(
                    "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors",
                    active
                      ? "bg-zinc-800 text-zinc-100"
                      : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200",
                  )}
                >
                  <Icon className="size-4" />
                  <span className="hidden sm:inline">{item.label}</span>
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
      <footer className="mx-auto max-w-6xl px-4 pb-10 pt-4 text-xs text-zinc-600">
        <p>
          Paper trading only — Alpaca paper account. Signals: z-score mean
          reversion (MR) &amp; trend continuation (TC), gated by LLM validation.{" "}
          <a
            href="https://www.drewboynton.com/projects/llm-advisor"
            className="text-zinc-500 underline-offset-2 hover:text-zinc-300 hover:underline"
          >
            Project deep dive →
          </a>
        </p>
      </footer>
    </div>
  );
}

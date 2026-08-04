"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image";
import type { ReactNode } from "react";
import clsx from "clsx";
import { ThemeToggle } from "@/components/ThemeToggle";

const navItems = [
  { href: "/", label: "Overview" },
  { href: "/trades", label: "Trades" },
  { href: "/breakdowns", label: "Breakdowns" },
  { href: "/funnel", label: "Funnel" },
  ...(process.env.NEXT_PUBLIC_COMMAND_CENTER_ENABLED === "true"
    ? [{ href: "/command-center", label: "Command Center" }]
    : []),
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-line bg-[color-mix(in_srgb,var(--paper)_86%,transparent)] backdrop-blur-lg backdrop-saturate-150">
        <div className="mx-auto flex h-[62px] max-w-[1240px] items-center gap-7 px-5 sm:px-7">
          <Link
            href="/"
            aria-label="LLM Advisor home"
            className="flex shrink-0 items-center gap-2.5"
          >
            <Image
              src="/llm-advisor-mark.png"
              alt=""
              aria-hidden="true"
              width={32}
              height={32}
              priority
              className="size-8 object-contain"
            />
            <span className="hidden text-[14.5px] font-semibold tracking-[-0.012em] sm:inline">
              LLM Advisor
            </span>
          </Link>

          <nav
            aria-label="Primary"
            className="ml-auto flex min-w-0 items-center gap-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          >
            {navItems.map((item) => {
              const active =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={clsx(
                    "whitespace-nowrap rounded-full px-2.5 py-1.5 text-[13px] transition-colors sm:px-3 sm:text-[13.5px]",
                    active
                      ? "bg-sunk font-medium text-ink"
                      : "text-ink-2 hover:bg-sunk hover:text-ink",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-[1240px] px-5 pb-16 pt-9 sm:px-7">
        {children}
      </main>

      <footer className="mx-auto max-w-[1240px] px-5 sm:px-7">
        <div className="flex flex-wrap justify-between gap-5 border-t border-line py-6 text-[12px] text-ink-3">
          <p className="max-w-[70ch]">
            <strong className="font-semibold text-ink-2">
              Paper money. Real mistakes.
            </strong>{" "}
            Alpaca paper account — z-score mean reversion (MR) and trend
            continuation (TC), gated by LLM validation.
          </p>
          <p>
            <a
              href="https://www.drewboynton.com/projects/llm-advisor"
              className="border-b border-line-2 pb-px text-ink-2 transition-colors hover:text-ink"
            >
              Project deep dive →
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

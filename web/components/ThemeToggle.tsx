"use client";

import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

type Theme = "light" | "dark";

function readTheme(): Theme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.getAttribute("data-theme") === "dark"
    ? "dark"
    : "light";
}

export function ThemeToggle() {
  // The inline script in layout.tsx has already set the attribute; this syncs
  // React to whatever it decided, after hydration.
  const [theme, setTheme] = useState<Theme>("light");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setTheme(readTheme());
    setMounted(true);
  }, []);

  function toggle() {
    const next: Theme = readTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch {
      // Private-mode browsers: the toggle still works for this page view.
    }
    setTheme(next);
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      className="grid size-[34px] shrink-0 place-items-center rounded-full border border-line-2 text-ink-2 transition-colors hover:bg-card hover:text-ink"
    >
      {/* Before hydration we don't know the theme, so render nothing rather
          than the wrong icon. */}
      {mounted ? (
        theme === "dark" ? (
          <Sun className="size-[15px]" />
        ) : (
          <Moon className="size-[15px]" />
        )
      ) : (
        <span className="size-[15px]" />
      )}
    </button>
  );
}

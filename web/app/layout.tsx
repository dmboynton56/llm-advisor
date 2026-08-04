import type { Metadata } from "next";
import { IBM_Plex_Mono, Instrument_Sans } from "next/font/google";
import { AppShell } from "@/components/AppShell";
import "./globals.css";

const instrumentSans = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-instrument-sans",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "LLM Advisor | Ops Dashboard",
  description:
    "Operations dashboard for an LLM-validated options paper-trading system: account equity, trade performance, setup breakdowns, and execution funnel.",
};

// Runs before first paint so the stored theme never flashes the wrong surface.
const themeScript = `(function(){try{var s=localStorage.getItem("theme");var d=s==="dark"||(!s&&window.matchMedia("(prefers-color-scheme: dark)").matches);document.documentElement.setAttribute("data-theme",d?"dark":"light");}catch(e){document.documentElement.setAttribute("data-theme","light");}})();`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme="light" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body
        suppressHydrationWarning
        className={`${instrumentSans.variable} ${plexMono.variable} min-h-screen bg-paper font-sans text-[14px] leading-normal text-ink antialiased`}
      >
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}

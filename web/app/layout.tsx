import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { AppShell } from "@/components/AppShell";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "LLM Advisor | Ops Dashboard",
  description:
    "Operations dashboard for an LLM-validated options paper-trading system: account equity, trade performance, setup breakdowns, and execution funnel.",
  icons: {
    icon: "/llm-advisor-mark.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        suppressHydrationWarning
        className={`${inter.className} min-h-screen bg-zinc-950 text-zinc-100 antialiased`}
      >
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}

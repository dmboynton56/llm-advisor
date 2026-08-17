import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import {
  COMMAND_CENTER_COOKIE,
  commandCenterToken,
} from "@/lib/commandCenterAuth";
import {
  alpacaPaperErrorPayload,
  fetchAlpacaPaperSnapshot,
} from "@/lib/alpacaPaper";
import { getLiveState } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  if (process.env.COMMAND_CENTER_ENABLED !== "true") {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }

  const password = process.env.COMMAND_CENTER_PASSWORD;
  if (!password) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const jar = await cookies();
  const cookie = jar.get(COMMAND_CENTER_COOKIE)?.value;
  if (!cookie || cookie !== commandCenterToken(password)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  if (process.env.ALPACA_PAPER_TRADING?.trim().toLowerCase() !== "true") {
    return NextResponse.json(
      {
        error:
          "ALPACA_PAPER_TRADING must be true — live blotter refuses non-paper keys",
      },
      { status: 500 },
    );
  }

  const key = (process.env.ALPACA_API_KEY ?? "").trim();
  const secret = (process.env.ALPACA_SECRET_KEY ?? "").trim();
  if (!key || !secret) {
    return NextResponse.json(
      { error: "ALPACA_API_KEY / ALPACA_SECRET_KEY not configured" },
      { status: 500 },
    );
  }

  const liveState = await getLiveState("paper");
  try {
    const payload = await fetchAlpacaPaperSnapshot(liveState);
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "alpaca fetch failed";
    return NextResponse.json(alpacaPaperErrorPayload(liveState, message), {
      status: 500,
    });
  }
}

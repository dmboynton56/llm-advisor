import {
  createHmac,
  randomBytes,
  timingSafeEqual,
} from "node:crypto";
import { asJsonRecord, firstJsonString, jsonNumber } from "@/lib/json";
import type { JsonValue } from "@/lib/types";

const TOKEN_VERSION = 1;
const DEFAULT_TOKEN_TTL_SECONDS = 12 * 60 * 60;

type DemoTokenPayload = {
  v: number;
  typ: "mobile-demo";
  sub: "private-paper-account";
  scope: "alpaca-paper-read";
  iat: number;
  exp: number;
  jti: string;
};

export type MobileDemoSession = {
  subject: DemoTokenPayload["sub"];
  scope: DemoTokenPayload["scope"];
  expiresAt: Date;
  tokenId: string;
};

function base64UrlEncode(value: string): string {
  return Buffer.from(value, "utf8")
    .toString("base64")
    .replace(/=/g, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
}

function base64UrlDecode(value: string): string | null {
  try {
    const padded = value.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(value.length / 4) * 4, "=");
    return Buffer.from(padded, "base64").toString("utf8");
  } catch {
    return null;
  }
}

function secret(): string | null {
  const value = process.env.MOBILE_DEMO_TOKEN_SECRET?.trim();
  return value || null;
}

function pairingCode(): string | null {
  const value = process.env.MOBILE_DEMO_PAIRING_CODE?.trim();
  return value || null;
}

function tokenTtlSeconds(): number {
  const configured = Number(process.env.MOBILE_DEMO_TOKEN_TTL_SECONDS ?? DEFAULT_TOKEN_TTL_SECONDS);
  if (!Number.isFinite(configured)) return DEFAULT_TOKEN_TTL_SECONDS;
  return Math.min(Math.max(Math.floor(configured), 15 * 60), 24 * 60 * 60);
}

function enabled(): boolean {
  return (
    process.env.MOBILE_DEMO_MODE === "true" &&
    process.env.ALPACA_PAPER_TRADING === "true"
  );
}

function digest(value: string): Buffer {
  return createHmac("sha256", secret() ?? "missing-mobile-demo-secret")
    .update(value)
    .digest();
}

function equalSecrets(left: string, right: string): boolean {
  const leftDigest = digest(left);
  const rightDigest = digest(right);
  return timingSafeEqual(leftDigest, rightDigest);
}

function sign(encodedPayload: string): string {
  return base64UrlEncode(digest(encodedPayload).toString("base64"));
}

export function mobileDemoEnabled(): boolean {
  return enabled() && Boolean(secret()) && Boolean(pairingCode());
}

export function pairingCodeMatches(value: string): boolean {
  const configured = pairingCode();
  if (!enabled() || !secret() || !configured) return false;
  return equalSecrets(value.trim(), configured);
}

export function issueMobileDemoToken(): {
  accessToken: string;
  expiresAt: Date;
} | null {
  if (!mobileDemoEnabled()) return null;

  const now = Math.floor(Date.now() / 1000);
  const payload: DemoTokenPayload = {
    v: TOKEN_VERSION,
    typ: "mobile-demo",
    sub: "private-paper-account",
    scope: "alpaca-paper-read",
    iat: now,
    exp: now + tokenTtlSeconds(),
    jti: randomBytes(18).toString("hex"),
  };
  const encodedPayload = base64UrlEncode(JSON.stringify(payload));
  const token = encodedPayload + "." + sign(encodedPayload);
  return { accessToken: token, expiresAt: new Date(payload.exp * 1000) };
}

function parseDemoTokenPayload(value: JsonValue): DemoTokenPayload | null {
  const record = asJsonRecord(value);
  if (!record) return null;
  const version = jsonNumber(record.v);
  const type = firstJsonString(record.typ);
  const subject = firstJsonString(record.sub);
  const scope = firstJsonString(record.scope);
  const issuedAt = jsonNumber(record.iat);
  const expiresAt = jsonNumber(record.exp);
  const tokenId = firstJsonString(record.jti);
  if (
    version !== TOKEN_VERSION ||
    type !== "mobile-demo" ||
    subject !== "private-paper-account" ||
    scope !== "alpaca-paper-read" ||
    issuedAt === null ||
    expiresAt === null ||
    tokenId === null
  ) {
    return null;
  }
  return {
    v: version,
    typ: type,
    sub: subject,
    scope,
    iat: issuedAt,
    exp: expiresAt,
    jti: tokenId,
  };
}

export function verifyMobileDemoToken(token: string): MobileDemoSession | null {
  if (!enabled() || !secret()) return null;
  const [encodedPayload, encodedSignature, extra] = token.split(".");
  if (!encodedPayload || !encodedSignature || extra) return null;

  const expectedSignature = sign(encodedPayload);
  if (!equalSecrets(encodedSignature, expectedSignature)) return null;

  const rawPayload = base64UrlDecode(encodedPayload);
  if (!rawPayload) return null;

  let payload: DemoTokenPayload | null;
  try {
    const parsed: JsonValue = JSON.parse(rawPayload);
    payload = parseDemoTokenPayload(parsed);
  } catch {
    return null;
  }
  if (!payload) return null;

  const now = Math.floor(Date.now() / 1000);
  if (
    payload.exp <= now ||
    payload.iat > now + 60
  ) {
    return null;
  }

  return {
    subject: payload.sub,
    scope: payload.scope,
    expiresAt: new Date(payload.exp * 1000),
    tokenId: payload.jti,
  };
}

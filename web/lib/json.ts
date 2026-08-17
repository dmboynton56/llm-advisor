import type { JsonRecord, JsonValue } from "@/lib/types";

export type JsonInput = JsonValue | undefined;

export function isJsonRecord(value: JsonInput): value is JsonRecord {
  return value !== undefined && value !== null && typeof value === "object" && !Array.isArray(value);
}

export function isJsonString(value: JsonInput): value is string {
  return typeof value === "string";
}

export function isJsonNumber(value: JsonInput): value is number {
  return typeof value === "number";
}

export function isJsonBoolean(value: JsonInput): value is boolean {
  return typeof value === "boolean";
}

export function asJsonRecord(value: JsonInput): JsonRecord | null {
  return isJsonRecord(value) ? value : null;
}

export function jsonRecords(value: JsonInput): JsonRecord[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const record = asJsonRecord(item);
    return record ? [record] : [];
  });
}

export function jsonString(value: JsonInput): string | null {
  if (!isJsonString(value)) return null;
  const text = value.trim();
  return text || null;
}

export function firstJsonString(...values: JsonInput[]): string | null {
  for (const value of values) {
    const text = jsonString(value);
    if (text) return text;
  }
  return null;
}

export function jsonNumber(value: JsonInput): number | null {
  if (isJsonNumber(value)) return Number.isFinite(value) ? value : null;
  const text = jsonString(value);
  if (!text) return null;
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

export function firstJsonNumber(...values: JsonInput[]): number | null {
  for (const value of values) {
    const number = jsonNumber(value);
    if (number !== null) return number;
  }
  return null;
}

export function jsonBoolean(value: JsonInput): boolean | null {
  return isJsonBoolean(value) ? value : null;
}

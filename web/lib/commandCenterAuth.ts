import { createHash } from "node:crypto";

export const COMMAND_CENTER_COOKIE = "llm_advisor_command_center";

export function commandCenterToken(password: string): string {
  return createHash("sha256")
    .update(`llm-advisor-command-center:${password}`)
    .digest("hex");
}

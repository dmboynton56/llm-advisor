"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import {
  COMMAND_CENTER_COOKIE,
  commandCenterToken,
} from "@/lib/commandCenterAuth";

export async function unlockCommandCenter(formData: FormData) {
  const configuredPassword = process.env.COMMAND_CENTER_PASSWORD;
  const suppliedPassword = String(formData.get("password") ?? "");

  if (!configuredPassword || suppliedPassword !== configuredPassword) {
    redirect("/command-center/login?error=1");
  }

  const jar = await cookies();
  jar.set(COMMAND_CENTER_COOKIE, commandCenterToken(configuredPassword), {
    httpOnly: true,
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production",
    path: "/command-center",
    maxAge: 60 * 60 * 8,
  });
  redirect("/command-center");
}

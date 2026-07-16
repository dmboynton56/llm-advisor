import { NextRequest, NextResponse } from "next/server";
import {
  COMMAND_CENTER_COOKIE,
  commandCenterToken,
} from "@/lib/commandCenterAuth";

export function proxy(request: NextRequest) {
  if (process.env.COMMAND_CENTER_ENABLED !== "true") return NextResponse.next();
  if (request.nextUrl.pathname === "/command-center/login") return NextResponse.next();

  const password = process.env.COMMAND_CENTER_PASSWORD;
  const cookie = request.cookies.get(COMMAND_CENTER_COOKIE)?.value;
  if (password && cookie === commandCenterToken(password)) return NextResponse.next();

  const login = request.nextUrl.clone();
  login.pathname = "/command-center/login";
  login.search = "";
  return NextResponse.redirect(login);
}

export const config = { matcher: ["/command-center/:path*"] };

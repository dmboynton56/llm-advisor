import { LockKeyhole } from "lucide-react";
import { notFound } from "next/navigation";
import { unlockCommandCenter } from "../actions";

export const dynamic = "force-dynamic";

export default async function CommandCenterLogin({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  if (process.env.COMMAND_CENTER_ENABLED !== "true") notFound();
  const { error } = await searchParams;
  const passwordConfigured = Boolean(process.env.COMMAND_CENTER_PASSWORD);

  return (
    <div className="mx-auto max-w-md rounded-xl border border-zinc-800 bg-zinc-900/60 p-6 shadow-2xl shadow-black/20">
      <div className="flex size-10 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-400">
        <LockKeyhole className="size-5" />
      </div>
      <h1 className="mt-4 text-xl font-semibold">Private command center</h1>
      <p className="mt-2 text-sm text-zinc-400">
        Enter the shared operator password. This lightweight gate is a scaffold;
        use identity-backed authentication before connecting a brokerage account.
      </p>
      <form action={unlockCommandCenter} className="mt-6 space-y-3">
        <label className="block text-xs font-medium uppercase tracking-wide text-zinc-500">
          Password
          <input
            type="password"
            name="password"
            required
            disabled={!passwordConfigured}
            className="mt-2 w-full rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-500"
          />
        </label>
        {error ? <p className="text-sm text-rose-400">Password not accepted.</p> : null}
        {!passwordConfigured ? (
          <p className="text-sm text-amber-400">COMMAND_CENTER_PASSWORD is not configured.</p>
        ) : null}
        <button
          type="submit"
          disabled={!passwordConfigured}
          className="w-full rounded-md bg-emerald-500 px-3 py-2 text-sm font-medium text-zinc-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Unlock
        </button>
      </form>
    </div>
  );
}

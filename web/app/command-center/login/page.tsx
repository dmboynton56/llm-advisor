import { LockKeyhole } from "lucide-react";
import { notFound } from "next/navigation";
import { Panel } from "@/components/ui";
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
    <Panel className="mx-auto max-w-md p-6">
      <div className="grid size-10 place-items-center rounded-panel border border-line-2 text-ink">
        <LockKeyhole className="size-5" />
      </div>
      <h1 className="mt-4 text-[22px] font-semibold tracking-[-0.022em]">
        Private command center
      </h1>
      <p className="mt-2 text-[13px] text-ink-2">
        Enter the shared operator password. This lightweight gate is a scaffold;
        use identity-backed authentication before connecting a brokerage account.
      </p>

      <form action={unlockCommandCenter} className="mt-6 flex flex-col gap-3">
        <label className="tag block">
          Password
          <input
            type="password"
            name="password"
            required
            disabled={!passwordConfigured}
            className="mt-2 w-full rounded-lg border border-line-2 bg-paper px-3 py-2 font-sans text-[13px] normal-case tracking-normal text-ink outline-none transition-colors focus:border-ink-3 disabled:opacity-50"
          />
        </label>

        {error ? (
          <p className="text-[13px] text-loss">Password not accepted.</p>
        ) : null}
        {!passwordConfigured ? (
          <p className="text-[13px] text-ink-2">
            COMMAND_CENTER_PASSWORD is not configured.
          </p>
        ) : null}

        <button
          type="submit"
          disabled={!passwordConfigured}
          className="w-full rounded-lg bg-ink px-3 py-2.5 text-[13px] font-medium text-paper transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Unlock
        </button>
      </form>
    </Panel>
  );
}

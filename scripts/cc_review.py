"""Review base runs with the Claude Code reviewer (a second reviewer family), billed to the Claude subscription.

    python3 scripts/cc_review.py --manifest runs.jsonl --results cc-results.jsonl --work-dir /data/cc-work \\
        [--env-file /path/.env] [--model claude-haiku-5-5] [--effort medium] [--concurrency 2] [--limit N] \\
        [--retry-failed] [--checker-version 1] [--max-steps 30] [--max-seconds 900] [--show-command]

The manifest lists one base run per line, its rebuilt state in a running container
(``taste.recovery_study.cc_review_driver``). The subscription's token is read from CLAUDE_CODE_OAUTH_TOKEN in the
environment, else from that one line of --env-file; it is never printed. --show-command prints the command line one
review would run, with placeholders, and runs nothing.

Exit status: 0 every listed run has a record; 1 some are left (failed, or --limit); 4 the job stopped (a usage,
spend or rate limit, the login, or the confinement): read stop.json beside the results, and start again only once
the cause is gone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.recovery_study.cc_review_driver import drive, load_manifest, verdicts
from taste.recovery_study.cc_reviewer import (
    ReviewSettings,
    Runner,
    allow_rule,
    command_line,
    read_token,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--env-file")
    parser.add_argument("--model", default=ReviewSettings.model)
    parser.add_argument("--effort", default=ReviewSettings.effort)
    parser.add_argument("--checker-version", default=ReviewSettings.checker_version)
    parser.add_argument("--max-steps", type=int, default=ReviewSettings.max_steps)
    parser.add_argument("--max-seconds", type=float, default=ReviewSettings.max_seconds)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--show-command", action="store_true")
    args = parser.parse_args(argv)
    settings = ReviewSettings(model=args.model, effort=args.effort, checker_version=args.checker_version,
                              max_steps=args.max_steps, max_seconds=args.max_seconds, claude=args.claude,
                              docker=args.docker)
    items = load_manifest(args.manifest)
    if args.show_command:
        paths = {"settings": "<work>/settings.json", "system": "<work>/system.txt"}
        first = items[0] if items else {"container": "<container>", "cwd": "<cwd>"}
        line = command_line(settings, paths, container="<container>", cwd="<cwd>", user=None, session="<uuid>")
        print(json.dumps({"argv": line, "allow": allow_rule("<container>", "<cwd>", None, settings.docker),
                          "first_run": first.get("run"), "runs": len(items)}, indent=1))
        return 0
    runner = Runner(read_token(args.env_file), settings)
    summary = drive(items, args.results, args.work_dir, runner, concurrency=args.concurrency,
                    retry_failed=args.retry_failed, limit=args.limit)
    summary["verdicts"] = verdicts(args.results)
    print(json.dumps(summary, indent=1))
    if summary["stopped"]:
        return 4
    return 0 if summary["verdicts"]["failed"] == 0 and summary["left"] == 0 and \
        sum(summary["verdicts"].values()) >= len(items) else 1


if __name__ == "__main__":
    sys.exit(main())

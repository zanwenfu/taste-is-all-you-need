"""Restore only the four records a replay needs of each named trial, from the parts of a trial archive.

    python3 -I scripts/restore_trial_records.py --tokens tokens.txt --out /data/taste-trials \\
        --parts /data/replay/archive/nov7-run [--set trials] [--index trials.files.tsv] [--sums SHA256SUMS] \\
        [--dry-run] [--json account.json]
    rclone cat gdrive:<archive>/trials.tar.zst.part-000 | python3 -I scripts/restore_trial_records.py \\
        --tokens tokens.txt --out /data/taste-trials --stdin [--index trials.files.tsv]

The tokens file lists one trial token per line (``scripts/recovery_replay.py --tokens-out`` writes it). Each set
named with ``--set`` (a slice such as ``trials-s03``, repeatable; default ``trials``) is read from its parts in
``--parts``; with the set's file list (``<set>.files.tsv``, found beside the parts or given with ``--index``) only the
sets that hold a wanted trial are read, and each only as far as its last wanted file. The parts are checked against
``SHA256SUMS`` first when it is beside them (``--no-sums`` skips it). ``--stdin`` reads one set's compressed stream
from standard input instead (the parts concatenated). Writes only under ``--out``
(``taste.recovery_study.archive_restore``). ``--dry-run`` reads the file lists alone and says what would be read
and written. Exit status 0 when every trial has its journal, ledger and settled record; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.recovery_study import archive_restore as restore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=Path, required=True, help="the trial tokens to restore, one per line")
    parser.add_argument("--out", type=Path, required=True, help="the trials root to restore them under")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parts", type=Path, help="the folder holding the sets' parts and file lists")
    source.add_argument("--stdin", action="store_true", help="read one set's compressed stream from standard input")
    parser.add_argument("--set", action="append", dest="sets", help="a set's name (default: trials); repeatable")
    parser.add_argument("--index", type=Path, help="the set's file list, when it is not beside the parts")
    parser.add_argument("--sums", type=Path, help="SHA256SUMS, when it is not beside the parts")
    parser.add_argument("--no-sums", action="store_true", help="do not check the parts against SHA256SUMS")
    parser.add_argument("--dry-run", action="store_true", help="read only the file lists; write nothing")
    parser.add_argument("--json", type=Path, help="also write the account here")
    arguments = parser.parse_args(argv)
    tokens = restore.tokens_from(arguments.tokens)
    sets = arguments.sets or ["trials"]
    if arguments.stdin and (len(sets) > 1 or (arguments.dry_run and not arguments.index)):
        parser.error("--stdin reads one set; a dry run of it needs --index")
    if arguments.index and len(sets) > 1:
        parser.error("--index names one set's file list")
    account = {"tokens": len(tokens), "sets": {}}
    for name in sets:
        index = arguments.index or (arguments.parts / f"{name}.files.tsv" if arguments.parts else None)
        planned = restore.plan(restore.read_index(index), tokens) if index and index.is_file() else None
        if planned is not None and not planned["members"]:
            account["sets"][name] = {"skipped": "holds none of the trials"}
            continue
        entry = {"index": str(index) if planned else None}
        if planned:
            entry.update(members=len(planned["members"]), bytes=planned["bytes"], trials=len(planned["tokens"]),
                         last=planned["last"][1] if planned["last"] else None)
        if arguments.dry_run:
            if planned is None:
                entry["note"] = "no file list: the whole set would be read"
            account["sets"][name] = entry
            continue
        if arguments.stdin:
            with restore.decompressed(stream=sys.stdin.buffer) as stream:
                found = restore.extract(stream, tokens, arguments.out,
                                        members=planned and planned["members"])
        else:
            paths = restore.parts(arguments.parts, name)
            digests = arguments.sums or arguments.parts / "SHA256SUMS"
            if not arguments.no_sums:
                if not digests.is_file():
                    parser.error(f"no SHA256SUMS at {digests} (give --sums or --no-sums)")
                entry["parts_checked"] = restore.check_parts(paths, restore.sums(digests))
            with restore.decompressed(paths=paths) as stream:
                found = restore.extract(stream, tokens, arguments.out, members=planned and planned["members"])
        entry.update({key: found[key] for key in ("written", "present", "conflicts", "refused", "read_members",
                                                   "stopped_early", "several_journals")})
        account["sets"][name] = entry
    missing = restore.on_disk(arguments.out, tokens) if not arguments.dry_run else None
    if missing is not None:
        account["whole"] = len(tokens) - len(missing)
        account["missing"] = missing
    if arguments.json:
        arguments.json.write_text(json.dumps(account, indent=1, sort_keys=True) + "\n")
    for name, entry in account["sets"].items():
        print(name, json.dumps({key: value for key, value in entry.items() if key not in ("conflicts", "refused")}))
        for key in ("conflicts", "refused"):
            for item in entry.get(key) or ():
                print(f"  {key}: {item}")
    if missing is not None:
        print(f"trials whole: {account['whole']} of {len(tokens)}")
        for token, kinds in sorted(missing.items())[:50]:
            print(f"  missing {token}: {', '.join(kinds)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

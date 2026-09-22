"""Store/worker shutdown releases Git pipes without relying on garbage collection."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="Linux descriptor accounting")


def descriptors_after(tmp_path, scenario):
    # Isolate the measurement from pytest's own descriptors and prior test
    # objects. Retained handles model a service keeping its completed history.
    script = """
import gc
import json
import os
import sys
from pathlib import Path
from taste.memstore import Store
gc.disable()
def descriptors():
    return len(os.listdir('/proc/self/fd'))
root = Path(sys.argv[1]) / 'repo'
""" + textwrap.dedent(scenario)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_closed_workers_release_git_pipes_with_retained_history(tmp_path):
    counts = descriptors_after(tmp_path, """
        from taste.brains.contract import Contract
        from taste.brains.subbrain import SubBrain
        store = Store.open(root, 'resources')
        retained = []
        for index in range(13):
            brain = SubBrain(store, Contract(identity='worker', task='write result',
                             outputs=('result.txt',), success_criteria=('result exists',)))
            brain.install_contract()
            brain.branch.write('result.txt', str(index))
            brain.checkpoint('completed work')
            brain.close()
            retained.append(brain)
            if index == 0:
                before = descriptors()
        after = descriptors()
        store.close()
        print(json.dumps({'before': before, 'after': after}))
    """)
    assert counts["after"] - counts["before"] <= 2, counts


def test_store_closes_released_branch_backends(tmp_path):
    counts = descriptors_after(tmp_path, """
        before = descriptors()
        store = Store.open(root, 'resources')
        retained = []
        for index in range(4):
            branch = store.branch('worker-' + str(index))
            branch.write('result.txt', str(index))
            branch.checkpoint('completed work')
            branch.release()
            assert branch.read('result.txt') == str(index)
            retained.append(branch)
        store.close()
        print(json.dumps({'before': before, 'after': descriptors()}))
    """)
    assert counts["after"] - counts["before"] <= 2, counts

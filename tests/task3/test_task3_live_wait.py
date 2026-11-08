"""task3_ttc.live_status --wait against the Kaggle cell-6 background wrapper (no model): done file only on exit 0,
exit code written on any exit, and an immediate stop with the log tail when the process is gone without a done file."""
import os
import subprocess
import sys
import time

from common.runinfo import REPO_ROOT

# Same wrapper shape as notes/TASK3_KAGGLE_CELLS.md cell 6, with the python command replaced by INNER.
WRAPPER = ('nohup bash -c "{inner} 2>&1 | tee -a {log}; rc=\\${{PIPESTATUS[0]}}; echo \\$rc > {exit}; '
           'if [ \\"\\$rc\\" = 0 ]; then echo 0 > {done}; fi" > /dev/null 2>&1 &\necho $! > {pid}\n')


def _files(tmp):
    return {k: str(tmp / f"t.{k}") for k in ("log", "done", "exit", "pid")}


def _launch(tmp, inner):
    f = _files(tmp)
    (tmp / "run").mkdir(exist_ok=True)
    subprocess.run(["bash", "-c", WRAPPER.format(inner=inner, **f)], check=True)
    return f


def _wait(tmp, f, every="600", poll="0.2"):
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "task3_ttc.live_status", "--wait", "--run", str(tmp / "run"),
                        "--log", f["log"], "--done", f["done"], "--exit", f["exit"], "--pid", f["pid"],
                        "--every", every, "--poll", poll], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    return p, time.time() - t


def test_success_writes_done_and_exit(tmp_path):
    f = _launch(tmp_path, "sleep 1; echo finished-ok")
    p, dt = _wait(tmp_path, f)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "done file: exit code 0" in p.stdout and dt < 60
    assert open(f["done"]).read().strip() == "0" and open(f["exit"]).read().strip() == "0"


def test_nonzero_exit_no_done_file_stops_with_tail(tmp_path):
    inner = "for i in \\$(seq 1 40); do echo line\\$i; done; sleep 1; exit 3"
    f = _launch(tmp_path, f"bash -c '{inner}'")
    p, dt = _wait(tmp_path, f)
    assert p.returncode == 1, p.stdout + p.stderr
    assert not os.path.exists(f["done"]) and open(f["exit"]).read().strip() == "3"
    assert "no longer running and there is no done file" in p.stdout and "exit code: 3" in p.stdout
    tail = p.stdout.split("--- last 30 lines")[1].splitlines()[1:]
    assert tail == [f"line{i}" for i in range(11, 41)]
    assert dt < 60  # well before the first 600 s report interval ends


def test_killed_wrapper_stops_without_exit_code(tmp_path):
    f = _launch(tmp_path, "sleep 307")
    time.sleep(0.5)
    os.kill(int(open(f["pid"]).read()), 9)  # wrapper first, so it cannot write the exit file
    try:
        p, dt = _wait(tmp_path, f)
    finally:
        subprocess.run(["pkill", "-9", "-f", "sleep 307"])
    assert p.returncode == 1 and "none written (killed?)" in p.stdout and dt < 60, p.stdout + p.stderr
    assert not os.path.exists(f["done"]) and not os.path.exists(f["exit"])


def test_missing_pid_file_stops(tmp_path):
    f = _files(tmp_path)
    (tmp_path / "run").mkdir()
    p, _ = _wait(tmp_path, f)
    assert p.returncode == 1 and "no pid file" in p.stdout


def test_done_appears_while_running_is_detected_between_reports(tmp_path):
    f = _launch(tmp_path, "sleep 2")
    p, dt = _wait(tmp_path, f, every="600", poll="0.2")
    assert p.returncode == 0 and dt < 60 and p.stdout.count("questions 0/50") >= 1, p.stdout

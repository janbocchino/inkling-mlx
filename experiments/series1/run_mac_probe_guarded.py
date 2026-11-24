import json
from pathlib import Path
import subprocess
import time
import sys

root = Path(__file__).resolve().parents[1]
context_run = "--context" in sys.argv
tuning_run = "--tuning" in sys.argv
stem = "mac-tuning" if tuning_run else "mac-context" if context_run else "mac-probe"
report_name = "MAC_TUNING_RESULTS.json" if tuning_run else "MAC_CONTEXT_RESULTS.json" if context_run else "MAC_INFERENCE_RESULTS.json"
report_path = root / "outputs" / report_name
stdout_path = root / "work" / (stem + ".out")
stderr_path = root / "work" / (stem + ".err")
command = [str(root / "work/validation-env/bin/python"), "-u", str(root / "outputs/mac_probe.py"),
           "--model", str(root / "outputs/Inkling-2-Mini-expert-2bit-g64"),
           "--report", str(report_path), "--max-tokens", "192"]
if context_run:
    command += ["--cases", "", "--context-lengths", "512,2048,4096"]
if tuning_run:
    command += ["--cases", "", "--context-lengths", "512,4096,4096", "--prefill-step", "256"]
started = time.monotonic()
reason = None
with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=root)
    critical_since = None
    while process.poll() is None:
        value = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                               capture_output=True, text=True).stdout.strip()
        if value == "4":
            critical_since = critical_since or time.monotonic()
        else:
            critical_since = None
        if critical_since and time.monotonic() - critical_since > 20:
            reason = "Stopped after more than 20 seconds of critical system memory pressure"
            break
        if time.monotonic() - started > 600:
            reason = "Stopped at the 10 minute test deadline"
            break
        time.sleep(2)
    if reason:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    code = process.wait()
data = json.loads(report_path.read_text()) if report_path.exists() else {}
data["process_exit_code"] = code
data["watchdog_elapsed_seconds"] = time.monotonic() - started
if reason:
    data["state"] = "stopped_by_resource_guard"
    data["watchdog"] = reason
report_path.write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps({"exit_code": code, "state": data.get("state"), "watchdog": reason,
                  "completed_cases": len(data.get("cases", []))}, indent=2))

from subprocess import TimeoutExpired
from tools.timeout import Timeout
import tools.process as process
import tempfile


def _parse_result(stdout: str):
    # Parse ProbLog output
    result = {}
    for line in stdout.splitlines():
        if ":" in line:
            spl = line.split(":")
            result[str(spl[0].strip())] = float(spl[1].strip())
    return result


def problog_run(problog_code: str, timeout: float):
    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as f:
        f.write(problog_code)
        filename = f.name

    # Launch subprocess
    proc = process.start_subprocess(["problog", filename])
    try:

        stdout, stderr = proc.communicate(timeout=timeout)
        return _parse_result(stdout)

    except TimeoutExpired:
        process.terminate(proc)
        return Timeout(timeout)

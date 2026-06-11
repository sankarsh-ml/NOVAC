import subprocess
import json
from pathlib import Path

def analyze_tampering(image_path):

    backend_dir = Path(__file__).resolve().parents[2]
    project_root = backend_dir.parent
    mvss_python = project_root / "mvss_venv" / "Scripts" / "python.exe"
    mvss_script = backend_dir / "MVSS-Net" / "mvss_predict.py"

    result = subprocess.run(
        [
            str(mvss_python),
            str(mvss_script),
            str(Path(image_path).resolve())
        ],
        capture_output=True,
        text=True,
        cwd=str(backend_dir),
        timeout=120
    )

    print("STDOUT:")
    print(repr(result.stdout))

    print("STDERR:")
    print(repr(result.stderr))

    if result.returncode != 0:
        raise Exception(result.stderr)

    return json.loads(result.stdout)

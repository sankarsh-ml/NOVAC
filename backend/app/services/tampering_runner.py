import subprocess
import json

def analyze_tampering(image_path):

    result = subprocess.run(
        [
            r"D:\novac\mvss_venv\Scripts\python.exe",
            r"D:\novac\backend\MVSS-Net\mvss_predict.py",
            image_path
        ],
        capture_output=True,
        text=True
    )

    print("STDOUT:")
    print(repr(result.stdout))

    print("STDERR:")
    print(repr(result.stderr))

    if result.returncode != 0:
        raise Exception(result.stderr)

    return json.loads(result.stdout)
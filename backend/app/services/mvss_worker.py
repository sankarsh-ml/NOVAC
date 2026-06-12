import contextlib
import json
import sys
import traceback
from pathlib import Path


class _StdoutToStderr:
    def write(self, value):
        sys.stderr.write(value)
        sys.stderr.flush()

    def flush(self):
        sys.stderr.flush()


def _send(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main():
    backend_dir = Path(__file__).resolve().parents[2]

    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    with contextlib.redirect_stdout(_StdoutToStderr()):
        from app.services.tampering_service import tampering_service

    _send({"ready": True, "worker": "mvss"})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            image_path = request["image_path"]

            with contextlib.redirect_stdout(_StdoutToStderr()):
                result = tampering_service.analyze(image_path)

            _send({"ok": True, "result": result})

        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _send({
                "ok": False,
                "error": str(exc)
            })


if __name__ == "__main__":
    main()

from pathlib import Path

pairs = [
    (Path("/tmp/infer_server.py"), Path("/tmp/infer_server.lf.py")),
    (Path("/tmp/start_infer_sidecar.sh"), Path("/tmp/start_infer_sidecar.lf.sh")),
    (Path("/tmp/server.py"), Path("/tmp/server.lf.py")),
    (Path("/tmp/restart_infer.sh"), Path("/tmp/restart_infer.lf.sh")),
]
for src, dst in pairs:
    if src.is_file():
        dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        print(src, "->", dst, dst.stat().st_size)

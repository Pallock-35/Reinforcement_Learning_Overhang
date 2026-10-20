import socket
import numpy as np
from senxor.mi48 import MI48
from senxor.utils import connect_senxor

HOST = "0.0.0.0"
PORT = 5001


def _try_mi48_call(device: MI48, method_name: str, *args, **kwargs) -> bool:
    method = getattr(device, method_name, None)
    if not callable(method):
        return False
    try:
        method(*args, **kwargs)
        return True
    except Exception:
        return False


def configure_optional_filtering(device: MI48) -> list[str]:
    enabled_steps: list[str] = []

    if _try_mi48_call(device, "disable_filter", f1=True, f2=True, f3=True):
        enabled_steps.append("disable_filter")
    if _try_mi48_call(device, "set_filter_1", 85):
        enabled_steps.append("set_filter_1")
    if _try_mi48_call(device, "enable_filter", f1=True, f2=False, f3=False, f3_ks_5=False):
        enabled_steps.append("enable_filter")
    if _try_mi48_call(device, "set_offset_corr", 0.0):
        enabled_steps.append("set_offset_corr")
    if _try_mi48_call(device, "set_sens_factor", 100):
        enabled_steps.append("set_sens_factor")
    _try_mi48_call(device, "get_sens_factor")

    return enabled_steps

mi48, port, ports = connect_senxor()
if mi48 is None:
    raise RuntimeError("No MI48 device found. Check USB connection and retry.")

mi48.set_fps(15)
filter_steps = configure_optional_filtering(mi48)
if filter_steps:
    print(f"MI48 filtering configured: {', '.join(filter_steps)}")
else:
    print("MI48 filtering unavailable; streaming raw thermal frames")
mi48.start(stream=True, with_header=False)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

conn, addr = server.accept()

while True:
    data, header = mi48.read()

    if data is None:
        continue

    frame = data.astype(np.float32)  # (62, 80)

    payload = frame.tobytes()

    conn.sendall(np.array([len(payload)], dtype=np.uint32).tobytes())
    conn.sendall(payload)

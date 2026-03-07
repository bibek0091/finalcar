import os

# 1. Patch threadSemaphores.py
f1 = "src/data/Semaphores/threads/threadSemaphores.py"
if os.path.exists(f1):
    with open(f1, "r") as f: code = f.read()
    if "SO_REUSEADDR" not in code:
        code = code.replace("self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)",
                            "self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n            if hasattr(socket, 'SO_REUSEPORT'): self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)")
        code = code.replace("print(f\"[Semaphores] Failed to bind UDP: {e}\")",
                            "print(f\"[Semaphores] Failed to bind UDP: {e}\")\n            self.sock = None")
        with open(f1, "w") as f: f.write(code)
        print(f"Patched {f1}")

# 2. Patch threadTrafficCommunication.py
f2 = "src/data/TrafficCommunication/threads/threadTrafficCommunication.py"
if os.path.exists(f2):
    with open(f2, "r") as f: code = f.read()
    if "SO_REUSEADDR" not in code:
        code = code.replace("self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)",
                            "self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n            if hasattr(socket, 'SO_REUSEPORT'): self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)")
        code = code.replace("print(f\"[TrafficComm] Failed to bind UDP: {e}\")",
                            "print(f\"[TrafficComm] Failed to bind UDP: {e}\")\n            self.sock = None")
        with open(f2, "w") as f: f.write(code)
        print(f"Patched {f2}")

# 3. Patch processAutonomous.py
f3 = "src/autonomous/threads/processAutonomous.py"
if os.path.exists(f3):
    with open(f3, "r") as f: code = f.read()
    if "os.environ.get('DISPLAY')" not in code:
        code = code.replace('cv2.imshow("BFMC Semantic Brain", dbg_frame)\n            cv2.waitKey(1)',
                            "if os.environ.get('DISPLAY'):\n                cv2.imshow(\"BFMC Semantic Brain\", dbg_frame)\n                cv2.waitKey(1)")
        with open(f3, "w") as f: f.write(code)
        print(f"Patched {f3}")

print("All Pi-side fixes successfully applied!")

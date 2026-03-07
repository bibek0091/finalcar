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

# 3. Patch processAutonomous.py (UNCONDITIONALLY DISABLE IMSHOW)
f3 = "src/autonomous/threads/processAutonomous.py"
if os.path.exists(f3):
    with open(f3, "r") as f: code = f.read()
    # Remove previous conditional if it exists
    if "if os.environ.get('DISPLAY'):" in code:
        code = code.replace("if os.environ.get('DISPLAY'):\n                cv2.imshow(\"BFMC Semantic Brain\", dbg_frame)\n                cv2.waitKey(1)", "pass # Headless mode enforced natively by patch")
    # Comment entirely if original exists
    code = code.replace('cv2.imshow("BFMC Semantic Brain", dbg_frame)\n            cv2.waitKey(1)',
                        "pass # Headless mode enforced natively by patch")
    with open(f3, "w") as f: f.write(code)
    print(f"Patched {f3}")

# 4. Remove BAD Headless QT Bypass from main.py if present
f4 = "main.py"
if os.path.exists(f4):
    with open(f4, "r") as f: code = f.read()
    if 'os.environ["QT_QPA_PLATFORM"] = "offscreen"' in code:
        code = code.replace('os.environ["QT_QPA_PLATFORM"] = "offscreen"', '# removed qt_qpa to prevent plugin crash')
        with open(f4, "w") as f: f.write(code)
        print(f"Patched {f4}")

# 5. Remove BAD Headless QT Bypass from traffic_module.py if present
f5 = "src/dashboard/traffic_module.py"
if os.path.exists(f5):
    with open(f5, "r") as f: code = f.read()
    if 'os.environ["QT_QPA_PLATFORM"] = "offscreen"' in code:
        code = code.replace('os.environ["QT_QPA_PLATFORM"] = "offscreen"', '# removed qt_qpa to prevent plugin crash')
        with open(f5, "w") as f: f.write(code)
        print(f"Patched {f5}")

print("All Pi-side fixes successfully applied! OpenCV Headless enforced.")

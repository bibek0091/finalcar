"""
apply_fixes.py — Run this on the Raspberry Pi after 'git pull'
to patch files that Git submodules prevent from syncing.

Usage:
    cd ~/finalcar
    python apply_fixes.py
"""
import os, re

def patch_file(path, description, patch_fn):
    if not os.path.exists(path):
        print(f"  SKIP {path} (not found)")
        return
    with open(path, "r") as f:
        original = f.read()
    patched = patch_fn(original)
    if patched != original:
        with open(path, "w") as f:
            f.write(patched)
        print(f"  PATCHED {path} — {description}")
    else:
        print(f"  OK {path} — already patched")

# ─────────────────────────────────────────────────────────────────────────────
# 1. threadSemaphores: SO_REUSEADDR + self.sock = None on failure
# ─────────────────────────────────────────────────────────────────────────────
def patch_semaphores(code):
    if "SO_REUSEADDR" not in code:
        code = code.replace(
            "self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)",
            "self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
            "            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
            "            if hasattr(socket, 'SO_REUSEPORT'):\n"
            "                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)"
        )
    if "self.sock = None" not in code.split("Failed to bind")[1] if "Failed to bind" in code else "":
        code = code.replace(
            'print(f"[Semaphores] Failed to bind UDP: {e}")',
            'print(f"[Semaphores] Failed to bind UDP: {e}")\n'
            '            self.sock = None'
        )
    return code

patch_file("src/data/Semaphores/threads/threadSemaphores.py",
           "SO_REUSEADDR + null socket on failure", patch_semaphores)

# ─────────────────────────────────────────────────────────────────────────────
# 2. threadTrafficCommunication: SO_REUSEADDR + self.sock = None on failure
# ─────────────────────────────────────────────────────────────────────────────
def patch_traffic(code):
    if "SO_REUSEADDR" not in code:
        code = code.replace(
            "self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)",
            "self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
            "            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
            "            if hasattr(socket, 'SO_REUSEPORT'):\n"
            "                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)"
        )
    if "self.sock = None" not in code.split("Failed to bind")[1] if "Failed to bind" in code else "":
        code = code.replace(
            'print(f"[TrafficComm] Failed to bind UDP: {e}")',
            'print(f"[TrafficComm] Failed to bind UDP: {e}")\n'
            '            self.sock = None'
        )
    return code

patch_file("src/data/TrafficCommunication/threads/threadTrafficCommunication.py",
           "SO_REUSEADDR + null socket on failure", patch_traffic)

# ─────────────────────────────────────────────────────────────────────────────
# 3. processAutonomous: strip ALL cv2.imshow / cv2.waitKey lines
# ─────────────────────────────────────────────────────────────────────────────
def patch_autonomous(code):
    lines = code.split("\n")
    new_lines = []
    skip_block = False
    for line in lines:
        stripped = line.strip()
        # Remove any line that calls cv2.imshow or cv2.waitKey
        if "cv2.imshow" in stripped or "cv2.waitKey" in stripped:
            continue
        new_lines.append(line)
    result = "\n".join(new_lines)
    # Remove bad QT_QPA_PLATFORM if present
    result = result.replace('os.environ["QT_QPA_PLATFORM"] = "offscreen"', "")
    return result

patch_file("src/autonomous/threads/processAutonomous.py",
           "removed all cv2.imshow/waitKey calls", patch_autonomous)

# ─────────────────────────────────────────────────────────────────────────────
# 4. main.py: remove bad QT_QPA_PLATFORM override if present
# ─────────────────────────────────────────────────────────────────────────────
def patch_main(code):
    code = code.replace('os.environ["QT_QPA_PLATFORM"] = "offscreen"', "# headless qt removed")
    return code

patch_file("main.py", "removed QT_QPA_PLATFORM override", patch_main)

# ─────────────────────────────────────────────────────────────────────────────
# 5. traffic_module.py: remove bad QT_QPA_PLATFORM override if present
# ─────────────────────────────────────────────────────────────────────────────
def patch_traffic_module(code):
    code = code.replace('os.environ["QT_QPA_PLATFORM"] = "offscreen"', "# headless qt removed")
    return code

patch_file("src/dashboard/traffic_module.py", "removed QT_QPA_PLATFORM override", patch_traffic_module)

# ─────────────────────────────────────────────────────────────────────────────
print("\n✅ All Pi-side fixes successfully applied!")
print("   You can now run: python main.py")

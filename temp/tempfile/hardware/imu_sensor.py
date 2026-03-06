import threading
import time
import math
import sys

try:
    import smbus
    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False
    print("[IMU] smbus module not found. IMU telemetry will mock zeros.")

class IMUSensor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.yaw_deg = 0.0
        self.running = False
        self.bus = None
        
        self.BNO_ADDR = 0x28
        self.OPR_MODE = 0x3D
        self.CHIP_ID = 0x00
        self.QUAT_W_LSB = 0x20
        
        if _SMBUS_AVAILABLE:
            try:
                self.bus = smbus.SMBus(1)
            except Exception as e:
                print(f"[IMU] Failed to open smbus: {e}")
                
    def read8(self, reg):
        if not self.bus: return 0
        return self.bus.read_byte_data(self.BNO_ADDR, reg)

    def write8(self, reg, value):
        if not self.bus: return
        self.bus.write_byte_data(self.BNO_ADDR, reg, value)

    def read16(self, reg):
        if not self.bus: return 0
        lsb = self.bus.read_byte_data(self.BNO_ADDR, reg)
        msb = self.bus.read_byte_data(self.BNO_ADDR, reg + 1)
        value = (msb << 8) | lsb
        if value > 32767:
            value -= 65536
        return value

    def run(self):
        if not self.bus:
            print("[IMU] Hardware inactive, thread stopping.")
            return
            
        print("[IMU] Waiting for sensor boot...")
        time.sleep(1)

        try:
            chip = self.read8(self.CHIP_ID)
            print("[IMU] Chip ID:", hex(chip))

            if chip != 0xA0:
                print("[IMU] BNO055 not detected!")
                return

            self.write8(self.OPR_MODE, 0x00)
            time.sleep(0.05)
            self.write8(self.OPR_MODE, 0x0C)
            time.sleep(0.1)
            print("[IMU] NDOF mode activated.")
            
            self.running = True
            while self.running:
                qw = self.read16(self.QUAT_W_LSB) / 16384.0
                qx = self.read16(self.QUAT_W_LSB + 2) / 16384.0
                qy = self.read16(self.QUAT_W_LSB + 4) / 16384.0
                qz = self.read16(self.QUAT_W_LSB + 6) / 16384.0

                yaw = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz)
                )

                self.yaw_deg = math.degrees(yaw)
                time.sleep(0.05)   # 20 Hz loop
        except Exception as e:
            print(f"[IMU] Exception in loop: {e}")
            self.running = False
            
    def stop(self):
        self.running = False
        
    def get_yaw(self):
        return self.yaw_deg

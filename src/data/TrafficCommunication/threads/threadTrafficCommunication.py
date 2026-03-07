import time
import socket
import json
import threading
from src.templates.threadwithstop import ThreadWithStop

class threadTrafficCommunication(ThreadWithStop):
    """
    Traffic Communication Thread
    Listens on UDP port 9000 for incoming GNSS/positional data from the BFMC server.
    Writes position and speed data to shared memory at 1 Hz.
    """
    def __init__(self, queuesList, pause=1.0):
        super(threadTrafficCommunication, self).__init__()
        self.queuesList = queuesList
        self._pause = pause
        
        # Shared Memory Keys required by BFMC
        self.key_speed = "deviceSpeed"
        self.key_pos = "devicePos"
        self.key_rot = "deviceRotation"
        self.key_history = "historyData"
        
        # State
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_rot = 0.0
        self.current_speed = 0.0
        
        self.udp_ip = "0.0.0.0"
        self.udp_port = 9000
        self.sock = None

    def _init_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Add REUSEPORT for Linux/Raspberry Pi
            if hasattr(socket, 'SO_REUSEPORT'):
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.sock.bind((self.udp_ip, self.udp_port))
            self.sock.setblocking(False)
            print(f"[TrafficComm] Listening on UDP {self.udp_port}")
        except Exception as e:
            print(f"[TrafficComm] Failed to bind UDP: {e}")
            self.sock = None

    def run(self):
        self._init_socket()
        super(threadTrafficCommunication, self).run()

    def thread_work(self):
        # 1. Read incoming UDP position data
        if self.sock:
            try:
                data, addr = self.sock.recvfrom(1024)
                if data:
                    msg = json.loads(data.decode('utf-8'))
                    if 'x' in msg and 'y' in msg:
                        self.current_x = float(msg['x'])
                        self.current_y = float(msg['y'])
                    # Emulate passing history or setting rot if available
            except BlockingIOError:
                pass
            except Exception as e:
                pass
                
        # 2. Write to Shared Memory at ~1Hz (driven by self._pause = 1.0)
        # BFMC Specification:
        # deviceSpeed -> [speed_cm_s]
        # devicePos -> [x_m, y_m]
        # deviceRotation -> [degrees_clockwise]
        # historyData -> [obstacle_id, x, y]
        
        if self.key_speed in self.queuesList:
            self.queuesList[self.key_speed].put([self.current_speed])
            
        if self.key_pos in self.queuesList:
            self.queuesList[self.key_pos].put([self.current_x, self.current_y])
            
        if self.key_rot in self.queuesList:
            self.queuesList[self.key_rot].put([self.current_rot])
            
        if self.key_history in self.queuesList:
            # Example: passing empty obstacle data or dummy placeholders
            self.queuesList[self.key_history].put([0, 0.0, 0.0])

    def stop(self):
        if self.sock:
            self.sock.close()
        super(threadTrafficCommunication, self).stop()

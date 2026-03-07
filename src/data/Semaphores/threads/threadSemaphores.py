import time
import socket
import json
import threading
from src.templates.threadwithstop import ThreadWithStop

class threadSemaphores(ThreadWithStop):
    """
    Semaphores Thread
    Listens on UDP port 5007 for incoming Semaphore (Traffic Light) states at 5 Hz.
    Writes the map of semaphore IDs to their states to Shared Memory.
    State representation: 0=RED, 1=YELLOW, 2=GREEN
    """
    def __init__(self, queuesList, pause=0.2):
        super(threadSemaphores, self).__init__()
        self.queuesList = queuesList
        self._pause = pause
        
        # Shared memory key
        self.key_semaphores = "semaphoresData"
        
        # Internal state map: { id_str : state_int }
        self.semaphore_states = {}
        
        self.udp_ip = "0.0.0.0"
        self.udp_port = 5007
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
            print(f"[Semaphores] Listening on UDP {self.udp_port}")
        except Exception as e:
            print(f"[Semaphores] Failed to bind UDP: {e}")
            self.sock = None

    def run(self):
        self._init_socket()
        super(threadSemaphores, self).run()

    def thread_work(self):
        # 1. Read incoming UDP semaphore data
        if self.sock:
            try:
                # Typically data might be sent somewhat rapidly, flush the buffer
                while True:
                    data, addr = self.sock.recvfrom(1024)
                    if data:
                        # Protocol assumed JSON format or BFMC specific delimited string
                        # Example: {"id": 1, "x": 1.2, "y": 3.4, "state": 2}
                        try:
                            msg = json.loads(data.decode('utf-8'))
                            if 'id' in msg and 'state' in msg:
                                s_id = str(msg['id'])
                                state = int(msg['state'])
                                self.semaphore_states[s_id] = state
                        except json.JSONDecodeError:
                            pass
            except BlockingIOError:
                pass # Socket buffer empty, proceed to write
            except Exception as e:
                pass
                
        # 2. Write Current States to Shared Memory at 5Hz (driven by pause=0.2)
        if self.key_semaphores in self.queuesList:
            # We copy the dictionary so the queue gets a snapshot
            self.queuesList[self.key_semaphores].put(self.semaphore_states.copy())

    def stop(self):
        if self.sock:
            self.sock.close()
        super(threadSemaphores, self).stop()

import sys
import time
import os
import psutil
import queue

# Create temp dir
os.makedirs("temp", exist_ok=True)
available_cores = list(range(psutil.cpu_count()))
psutil.Process(os.getpid()).cpu_affinity(available_cores)

sys.path.append(".")
from multiprocessing import Queue, Event
from src.utils.bigPrintMessages import BigPrint
from src.utils.outputWriters import QueueWriter, MultiWriter
import logging

logging.basicConfig(level=logging.INFO)

# ===================================== PROCESS IMPORTS ==================================
from src.hardware.camera.processCamera import processCamera

try:
    from src.data.Semaphores.processSemaphores import processSemaphores
except ModuleNotFoundError:
    try:
        from src.data.semaphores.processSemaphores import processSemaphores
    except ModuleNotFoundError:
        processSemaphores = None
        logging.error("V2X Semaphores MISSING. Traffic lights disabled!")

try:
    from src.data.TrafficCommunication.processTrafficCommunication import processTrafficCommunication
except ModuleNotFoundError:
    try:
        from src.data.trafficCommunication.processTrafficCommunication import processTrafficCommunication
    except ModuleNotFoundError:
        processTrafficCommunication = None
        logging.error("V2X TrafficCommunication MISSING. Telemetry disabled!")

from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.allMessages import StateChange
from src.statemachine.stateMachine import StateMachine
from src.statemachine.systemMode import SystemMode

# ===================================== SHUTDOWN PROCESS ====================================
def shutdown_process(process, timeout=1):
    process.join(timeout)
    if process.is_alive():
        print(f"The process {process} cannot normally stop, it's blocked somewhere! Terminate it!")
        process.terminate()
        process.join(timeout)
        if process.is_alive():
            print(f"The process {process} is still alive after terminate, killing it!")
            process.kill()
    print(f"The process {process} stopped")

def manage_process_life(process_class, process_instance, process_args, enabled, allProcesses):
    if process_class is None: return None
    if enabled:
        if process_instance is None:
            process_instance = process_class(*process_args)
            allProcesses.append(process_instance)
            process_instance.start()
    else:
        if process_instance is not None and process_instance.is_alive():
            shutdown_process(process_instance)
            allProcesses.remove(process_instance)
            process_instance = None
    return process_instance

# ===================================== TKINTER UI IMPORTS ==================================
import tkinter as tk
from PIL import Image, ImageTk
import base64
import numpy as np
import cv2

# Import map_engine and UI structure from the root-level modules
from map_engine import MapEngine
from dashboard_ui import DashboardUI

class BFMC_App:
    """Application Controller for the Python Tkinter Dashboard."""
    def __init__(self, root, qList):
        self.root = root
        self.root.title("BFMC ADAS Command Center")
        self.root.geometry("1400x850")
        self.qList = qList

        # Load Dashboard GUI Components
        self.ui = DashboardUI(self.root, self)
        self.map_engine = MapEngine()

        # Status variables
        self.car_x, self.car_y, self.car_yaw = 0.5, 0.5, 0.0
        self.current_speed, self.current_steer = 0.0, 0.0
        self.is_connected = True
        self.mode = "DRIVE"
        self.path = []
        self.visited_path_nodes = set()
        self.path_signs = []

        self.render_map()

    def set_mode(self, m):
        self.mode = m
        self.ui.var_main_mode.set(m)
        self.ui.log_event(f"Mode changed to: {m}")

    def on_map_click(self, event):
        canvas_x = self.ui.map_canvas.canvasx(event.x)
        canvas_y = self.ui.map_canvas.canvasy(event.y)
        mx, my = self.map_engine.to_meter(canvas_x, canvas_y)

        if self.mode == "DRIVE":
            self.car_x, self.car_y = mx, my
            self.render_map()

    def render_map(self):
        pil = self.map_engine.render_map(
            self.car_x, self.car_y, self.car_yaw,
            self.path, self.visited_path_nodes, self.path_signs,
            True, None, None, None
        )
        self.tk_map = ImageTk.PhotoImage(pil)
        self.ui.map_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_map)
        self.ui.map_canvas.config(scrollregion=self.ui.map_canvas.bbox(tk.ALL))

    def update_dashboard_ui(self, telemetry_data):
        self.ui.lbl_telemetry.config(
            text=f"SPD: {telemetry_data.get('speed_pwm', 0):.0f} | STR: {telemetry_data.get('steer_angle', 0):.1f}° | ERR: {telemetry_data.get('lane_error', 0):+.1f}px"
        )
        self.ui.lbl_ai.config(text=f"AI: {telemetry_data.get('state', 'OFF')}")

        # Helper to decode and display base64 -> OpenCV -> PIL
        def show_img_safe(b64_str, label_widget):
            if not b64_str: return
            try:
                img_data = base64.b64decode(b64_str)
                np_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cw, ch = label_widget.winfo_width(), label_widget.winfo_height()
                    img_pil = Image.fromarray(img_rgb).resize((cw if cw > 20 else 440, ch if ch > 20 else 330))
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    label_widget.imgtk = img_tk  # Keep reference
                    label_widget.configure(image=img_tk)
            except Exception as e:
                logging.debug(f"[UI] Frame decode error: {e}")

        show_img_safe(telemetry_data.get("yolo_b64", ""), self.ui.cam_label)
        show_img_safe(telemetry_data.get("bev_b64", ""), self.ui.bev_label)

    # Stub commands referenced by dashboard_ui.py
    def save_config(self): self.ui.log_event("Config Saved")
    def load_config(self): self.ui.log_event("Config Loaded")
    def toggle_connection(self): self.ui.log_event("Hardware connected internally via processAutonomous")
    def toggle_auto_mode(self): self.ui.log_event("Switched Mode")
    def toggle_adas_mode(self): self.ui.log_event("ADAS Toggled")
    def clear_route(self): pass


# ======================================== SETTING UP ====================================

def main():
    print(BigPrint.PLEASE_WAIT.value)
    allProcesses = list()
    allEvents = list()

    queueList = {
        "Critical":         Queue(),
        "Warning":          Queue(),
        "General":          Queue(),
        "Config":           Queue(),
        "Log":              Queue(),
        "Autonomous":       Queue(),
        "TkinterTelemetry": Queue(maxsize=2),
    }
    logger = logging.getLogger()

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    queue_writer = QueueWriter(queueList["Log"])
    sys.stdout = MultiWriter(original_stdout, queue_writer)
    sys.stderr = MultiWriter(original_stderr, queue_writer)

    # ===================================== INITIALIZE STATE ==================================
    stateChangeSubscriber = messageHandlerSubscriber(queueList, StateChange, "lastOnly", True)
    StateMachine.initialize_shared_state(queueList)

    # ===================================== INITIALIZE HARDWARE & V2X =======================
    camera_ready = Event()
    processCamera_inst = processCamera(queueList, logger, camera_ready, debugging=False)
    allProcesses.append(processCamera_inst)
    allEvents.append(camera_ready)

    semaphore_ready = Event()
    if processSemaphores is not None:
        processSemaphore_inst = processSemaphores(queueList, logger, semaphore_ready, debugging=False)
        allProcesses.append(processSemaphore_inst)
    else:
        processSemaphore_inst = None
        semaphore_ready.set()
    allEvents.append(semaphore_ready)

    traffic_com_ready = Event()
    if processTrafficCommunication is not None:
        processTrafficCom_inst = processTrafficCommunication(queueList, logger, 3, traffic_com_ready, debugging=False)
        allProcesses.append(processTrafficCom_inst)
    else:
        processTrafficCom_inst = None
        traffic_com_ready.set()
    allEvents.append(traffic_com_ready)

    # ===================================== INITIALIZE AUTONOMOUS ==========================
    from src.autonomous.threads.processAutonomous import processAutonomous
    autonomous_process = processAutonomous(queueList, logger)
    allProcesses.append(autonomous_process)

    # Start all background processes
    for process in allProcesses:
        process.daemon = True
        process.start()

    # ===================================== WAIT FOR HARDWARE READY ========================
    for event in allEvents:
        event.wait()

    StateMachine.initialize_starting_mode()

    # ===================================== TKINTER UI =====================================
    root = tk.Tk()
    app = BFMC_App(root, queueList)

    def poll_queues():
        nonlocal processSemaphore_inst, processTrafficCom_inst

        # 1. State Machine Polling
        try:
            message = stateChangeSubscriber.receive_nowait()
            if message is not None:
                modeDictSemaphore = SystemMode[message].value["semaphore"]["process"]
                modeDictTrafficCom = SystemMode[message].value["traffic_com"]["process"]

                processSemaphore_inst = manage_process_life(
                    processSemaphores, processSemaphore_inst,
                    [queueList, logger, semaphore_ready, False],
                    modeDictSemaphore["enabled"], allProcesses
                )
                processTrafficCom_inst = manage_process_life(
                    processTrafficCommunication, processTrafficCom_inst,
                    [queueList, logger, 3, traffic_com_ready, False],
                    modeDictTrafficCom["enabled"], allProcesses
                )
        except Exception as e:
            logging.debug(f"[StateMachine] Poll error: {e}")

        # 2. Telemetry Polling for Tkinter Dashboard
        telemetry_queue = queueList.get("TkinterTelemetry")
        try:
            if telemetry_queue is not None and not telemetry_queue.empty():
                telem_data = telemetry_queue.get_nowait()
                app.update_dashboard_ui(telem_data)
        except Exception as e:
            logging.debug(f"[UI] Telemetry poll error: {e}")

        root.after(50, poll_queues)

    # Start the scheduling loop
    root.after(100, poll_queues)

    print(BigPrint.C4_BOMB.value)
    print("Python Dashboard active. Close the window to stop the stack.")

    # Launch Main Thread UI Loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKeyboardInterruption caught!")

    print("\nShutting down all background processes...")
    for proc in reversed(allProcesses):
        try:
            proc.stop()
        except Exception:
            pass
    for proc in reversed(allProcesses):
        shutdown_process(proc)

if __name__ == "__main__":
    main()
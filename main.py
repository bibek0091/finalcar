# ===================================== GENERAL IMPORTS ==================================
import sys
import time
import os
import psutil

# Automatically create the 'temp' directory if it's missing to prevent FileNotFoundError
os.makedirs("temp", exist_ok=True)

# Pin to CPU cores 0–3 to maximize Raspberry Pi performance
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
from src.hardware.serialhandler.processSerialHandler import processSerialHandler

# BULLETPROOF LOADER: If folders are completely missing, don't crash.
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
    """Helper function to gracefully shutdown a process."""
    process.join(timeout)
    if process.is_alive():
        print(f"The process {process} cannot normally stop, it's blocked somewhere! Terminate it!")
        process.terminate()  # force terminate if it won't stop
        process.join(timeout)  # give it a moment to terminate
        if process.is_alive():
            print(f"The process {process} is still alive after terminate, killing it!")
            process.kill()  # last resort
    print(f"The process {process} stopped")

# ===================================== PROCESS MANAGEMENT ==================================
def manage_process_life(process_class, process_instance, process_args, enabled, allProcesses):
    """Start or stop a process based on the enabled flag (Controlled by BFMC State Machine)."""
    if process_class is None:
        return None  # Fail gracefully if module is missing
        
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

# ======================================== SETTING UP ====================================
print(BigPrint.PLEASE_WAIT.value)
allProcesses = list()
allEvents = list()

queueList = {
    "Critical": Queue(),
    "Warning": Queue(),
    "General": Queue(),
    "Config": Queue(),
    "Log": Queue(),
    "Vision": Queue(),       # Custom: Used by our YOLO script
    "Autonomous": Queue()    # Custom: Used by our Brain script
}
logging = logging.getLogger()

original_stdout = sys.stdout
original_stderr = sys.stderr

queue_writer = QueueWriter(queueList["Log"])
sys.stdout = MultiWriter(original_stdout, queue_writer)
sys.stderr = MultiWriter(original_stderr, queue_writer)

# ===================================== INITIALIZE ==================================
stateChangeSubscriber = messageHandlerSubscriber(queueList, StateChange, "lastOnly", True)
StateMachine.initialize_shared_state(queueList)

# ===================================== INITIALIZE HARDWARE & V2X =======================

# Initializing camera
camera_ready = Event()
processCamera_inst = processCamera(queueList, logging, camera_ready, debugging=False)
allProcesses.append(processCamera_inst)
allEvents.append(camera_ready)

# Initializing semaphores (UDP Traffic Lights)
semaphore_ready = Event()
if processSemaphores is not None:
    processSemaphore_inst = processSemaphores(queueList, logging, semaphore_ready, debugging=False)
    allProcesses.append(processSemaphore_inst)
else:
    processSemaphore_inst = None
    semaphore_ready.set() # Fake ready to prevent deadlock
allEvents.append(semaphore_ready)

# Initializing Traffic Communication (TCP LiveTraffic - The '3' is your Car ID)
traffic_com_ready = Event()
if processTrafficCommunication is not None:
    processTrafficCom_inst = processTrafficCommunication(queueList, logging, 3, traffic_com_ready, debugging=False)
    allProcesses.append(processTrafficCom_inst)
else:
    processTrafficCom_inst = None
    traffic_com_ready.set() # Fake ready to prevent deadlock
allEvents.append(traffic_com_ready)

# Initializing serial connection NUCLEO -> PI
serial_handler_ready = Event()
processSerialHandler_inst = processSerialHandler(queueList, logging, serial_handler_ready, debugging=False)
allProcesses.append(processSerialHandler_inst)
allEvents.append(serial_handler_ready)


# ===================================== INITIALIZE OUR CUSTOM STACK =====================

# 1. Start the Gateway (WebSockets for Telemetry to Dashboard)
try:
    from src.gateway.processGateway import processGateway
    gateway_process = processGateway(queueList, logging)
    allProcesses.append(gateway_process)
except Exception as e:
    logging.warning(f"Skipping Gateway: {e}")

# 2. Start the Vision Process (YOLO)
try:
    from src.autonomous.threads.processVision import processVision
    vision_process = processVision(queueList, logging)
    allProcesses.append(vision_process)
except Exception as e:
    logging.warning(f"Skipping Vision: {e}")

# 3. Start the Autonomous Process (The Brain)
from src.autonomous.threads.processAutonomous import processAutonomous
autonomous_process = processAutonomous(queueList, logging)
allProcesses.append(autonomous_process)

# Start all processes initially
for process in allProcesses:
    process.daemon = True
    process.start()

# ===================================== STAYING ALIVE ====================================

blocker = Event()
try:
    # wait for all hardware events to be set (Camera, Serial, etc.)
    for event in allEvents:
        event.wait()

    # apply starting mode
    StateMachine.initialize_starting_mode()

    time.sleep(10)
    print(BigPrint.C4_BOMB.value)
    print(BigPrint.PRESS_CTRL_C.value)

    while True:
        message = stateChangeSubscriber.receive()
        if message is not None:
            modeDictSemaphore = SystemMode[message].value["semaphore"]["process"]
            modeDictTrafficCom = SystemMode[message].value["traffic_com"]["process"]

            processSemaphore_inst = manage_process_life(processSemaphores, processSemaphore_inst, [queueList, logging, semaphore_ready, False], modeDictSemaphore["enabled"], allProcesses)
            processTrafficCom_inst = manage_process_life(processTrafficCommunication, processTrafficCom_inst, [queueList, logging, 3, traffic_com_ready, False], modeDictTrafficCom["enabled"], allProcesses)

        blocker.wait(0.1)

except KeyboardInterrupt:
    print("\nCatching a KeyboardInterruption exception! Shutdown all processes.\n")

    for proc in reversed(allProcesses):
        proc.stop()

    # wait for all processes to finish before exiting
    for proc in reversed(allProcesses):
        shutdown_process(proc)
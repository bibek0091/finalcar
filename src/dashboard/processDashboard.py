# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.
# [License text omitted for brevity - same BSD 3-Clause as original]

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

import queue
import psutil
import json
import inspect
import os
import time
import threading
import base64

from flask import Flask, request, send_from_directory, send_file
from flask_socketio import SocketIO
from flask_cors import CORS
from enum import Enum

from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.templates.workerprocess import WorkerProcess
from src.utils.messages.allMessages import Semaphores
from src.statemachine.stateMachine import StateMachine
from src.dashboard.components.calibration import Calibration
from src.dashboard.components.ip_manger import IpManager
from src.dashboard.graph_navigator import get_graph

import src.utils.messages.allMessages as allMessages


class processDashboard(WorkerProcess):
    """Dashboard process: serves web UI, handles SocketIO for BEV/YOLO/map/decisions."""

    def __init__(self, queueList, logging, ready_event=None, debugging=False):
        self.running = True
        self.queueList = queueList
        self.logger = logging
        self.debugging = debugging

        # IP replacement
        IpManager.replace_ip_in_file()

        # State machine
        self.stateMachine = StateMachine.get_instance()

        # Message handling
        self.messages = {}
        self.sendMessages = {}
        self.messagesAndVals = {}

        # Hardware monitoring
        self.memoryUsage = 0
        self.cpuCoreUsage = 0
        self.cpuTemperature = 0

        # Heartbeat
        self.heartbeat_last_sent = time.time()
        self.heartbeat_retries = 0
        self.heartbeat_max_retries = 3
        self.heartbeat_time_between_heartbeats = 20
        self.heartbeat_time_between_retries = 5
        self.heartbeat_received = False

        # Session management
        self.sessionActive = False
        self.activeUser = None

        # Serial connection state
        self.serialConnected = False

        # Configuration
        self.table_state_file = self._get_table_state_path()

        # ── Graph & Localization ─────────────────────────────────────
        self.graph = get_graph()
        self.current_route = []
        self.placed_signs = []  # [{label, node_id}]
        self.car_running = False

        # ── Flask + SocketIO ─────────────────────────────────────────
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        CORS(self.app, supports_credentials=True)

        # ── Routes ───────────────────────────────────────────────────
        dashboard_dir = os.path.dirname(__file__)
        frontend_dir = os.path.join(dashboard_dir, "frontend", "dist", "dashboard", "browser")

        @self.app.route('/')
        def serve_dashboard():
            return send_file(os.path.join(dashboard_dir, 'dashboard.html'))

        @self.app.route('/angular')
        def serve_angular():
            return send_from_directory(frontend_dir, 'index.html')

        @self.app.route('/<path:path>')
        def serve_static(path):
            try:
                return send_from_directory(frontend_dir, path)
            except Exception:
                return send_from_directory(frontend_dir, 'index.html')

        # Calibration
        self.calibration = Calibration(self.queueList, self.socketio)

        # Initialize message handling
        self._initialize_messages()
        self._setup_websocket_handlers()
        self._start_background_tasks()

        super(processDashboard, self).__init__(self.queueList, ready_event)

    def _get_table_state_path(self):
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_path, 'src', 'utils', 'table_state.json')

    def _initialize_messages(self):
        self.get_name_and_vals()
        self.messagesAndVals.pop("mainCamera", None)
        self.messagesAndVals.pop("Semaphores", None)
        self.subscribe()

    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers — both original and new dashboard."""
        # Original BFMC handlers
        self.socketio.on_event('message', self.handle_message)
        self.socketio.on_event('save', self.handle_save_table_state)
        self.socketio.on_event('load', self.handle_load_table_state)

        # ── New Dashboard Handlers ───────────────────────────────────
        @self.socketio.on('get_graph')
        def on_get_graph():
            self.socketio.emit('graph_data', self.graph.to_json())

        @self.socketio.on('find_route')
        def on_find_route(data):
            start = str(data.get('start', ''))
            end = str(data.get('end', ''))
            route = self.graph.find_route(start, end)
            self.current_route = route
            self.socketio.emit('route_result', {
                'route': route,
                'coords': self.graph.route_coords(route),
            })

        @self.socketio.on('place_sign')
        def on_place_sign(data):
            self.placed_signs.append({
                'label': data.get('label', ''),
                'node_id': str(data.get('node_id', '')),
            })

        @self.socketio.on('clear_signs')
        def on_clear_signs():
            self.placed_signs.clear()

        @self.socketio.on('start_car')
        def on_start_car():
            self.car_running = True
            # Send KL:30 via the state machine
            self.stateMachine.request_mode("dashboard_KL30_button")
            self.socketio.emit('log_line', '▶ Car started (KL:30)')

        @self.socketio.on('stop_car')
        def on_stop_car():
            self.car_running = False
            self.stateMachine.request_mode("dashboard_KL15_button")
            self.socketio.emit('log_line', '■ Car stopped (KL:15)')

    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        psutil.cpu_percent(interval=1, percpu=False)

        t1 = threading.Thread(target=self.update_hardware_data, daemon=True)
        t2 = threading.Thread(target=self.send_continuous_messages, daemon=True)
        t3 = threading.Thread(target=self.send_hardware_data_to_frontend, daemon=True)
        t4 = threading.Thread(target=self.send_heartbeat, daemon=True)
        t5 = threading.Thread(target=self.stream_console_logs, daemon=True)
        t6 = threading.Thread(target=self.stream_dashboard_feeds, daemon=True)
        for t in [t1, t2, t3, t4, t5, t6]:
            t.start()

    # ═══════════════════════════════════════════════════════════════════
    # New: Stream BEV, YOLO, and Decision data to the new dashboard
    # ═══════════════════════════════════════════════════════════════════
    def stream_dashboard_feeds(self):
        """Read from DashBEV, DashYOLO, DashDecision queues and emit to frontend."""
        while self.running:
            try:
                # BEV frame (base64 JPEG)
                bev_q = self.queueList.get("DashBEV")
                if bev_q:
                    try:
                        frame_b64 = bev_q.get_nowait()
                        self.socketio.emit('bev_frame', frame_b64)
                    except queue.Empty:
                        pass

                # YOLO frame (base64 JPEG)
                yolo_q = self.queueList.get("DashYOLO")
                if yolo_q:
                    try:
                        frame_b64 = yolo_q.get_nowait()
                        self.socketio.emit('yolo_frame', frame_b64)
                    except queue.Empty:
                        pass

                # Decision data (JSON dict)
                dec_q = self.queueList.get("DashDecision")
                if dec_q:
                    try:
                        decision = dec_q.get_nowait()
                        self.socketio.emit('decision', decision)

                        # Sign-based localization
                        labels = decision.get('yolo_labels', [])
                        for label in labels:
                            pos = self.graph.update_car_position(
                                label, self.placed_signs, self.current_route
                            )
                            if pos:
                                self.socketio.emit('car_position', pos)
                                self.socketio.emit('log_line',
                                    f'📍 Localized at node {pos["node_id"]} via {label}')
                    except queue.Empty:
                        pass

                time.sleep(0.05)  # ~20 fps max
            except Exception as e:
                if self.debugging:
                    self.logger.error(f"Dashboard feed error: {e}")
                time.sleep(1)

    def stream_console_logs(self):
        """Monitor the Log queue and emit messages to frontend."""
        log_queue = self.queueList.get("Log")
        if not log_queue:
            return

        while self.running:
            try:
                while not log_queue.empty():
                    msg = log_queue.get_nowait()
                    self.socketio.emit('console_log', {'data': msg})
                    self.socketio.emit('log_line', str(msg))
                    time.sleep(0)
                time.sleep(0.1)
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                if self.debugging:
                    self.logger.error(f"Error streaming logs: {e}")
                time.sleep(1)

    # ═══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════════════
    def _init_threads(self):
        """No threads needed — HTTP server runs in its own thread."""
        pass

    def stop(self):
        super(processDashboard, self).stop()
        self.running = False

    def run(self):
        if self.ready_event:
            self.ready_event.set()
        self.socketio.run(self.app, host='0.0.0.0', port=5005, allow_unsafe_werkzeug=True)

    # ═══════════════════════════════════════════════════════════════════
    # Original BFMC message infrastructure (preserved)
    # ═══════════════════════════════════════════════════════════════════
    def subscribe(self):
        for name, enum in self.messagesAndVals.items():
            if enum["owner"] != "Dashboard":
                subscriber = messageHandlerSubscriber(self.queueList, enum["enum"], "lastOnly", True)
                self.messages[name] = {"obj": subscriber}
            else:
                sender = messageHandlerSender(self.queueList, enum["enum"])
                self.sendMessages[str(name)] = {"obj": sender}
        subscriber = messageHandlerSubscriber(self.queueList, Semaphores, "fifo", True)
        self.messages["Semaphores"] = {"obj": subscriber}

    def get_name_and_vals(self):
        classes = inspect.getmembers(allMessages, inspect.isclass)
        for name, cls in classes:
            if name != "Enum" and issubclass(cls, Enum):
                self.messagesAndVals[name] = {"enum": cls, "owner": cls.Owner.value}

    def send_message_to_brain(self, dataName, dataDict):
        if dataName in self.sendMessages:
            self.sendMessages[dataName]["obj"].send(dataDict.get("Value"))

    def handle_message(self, data):
        if self.debugging:
            self.logger.info("Received message: " + str(data))
        try:
            dataDict = json.loads(data)
            dataName = dataDict["Name"]
            socketId = request.sid
            if dataName == "SessionAccess":
                self.handle_single_user_session(socketId)
            elif self.sessionActive and self.activeUser != socketId:
                return
            if dataName == "Heartbeat":
                self.handle_heartbeat()
            elif dataName == "SessionEnd":
                self.handle_session_end(socketId)
            elif dataName == "DrivingMode":
                self.handle_driving_mode(dataDict)
            elif dataName == "Calibration":
                self.handle_calibration(dataDict, socketId)
            elif dataName == "GetCurrentSerialConnectionState":
                self.handle_get_current_serial_connection_state(socketId)
            else:
                self.send_message_to_brain(dataName, dataDict)
            self.socketio.emit('response', {'data': 'Message received: ' + str(data)}, room=socketId)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON message: {e}")

    def handle_heartbeat(self):
        self.heartbeat_retries = 0
        self.heartbeat_last_sent = time.time()
        self.heartbeat_received = True

    def handle_driving_mode(self, dataDict):
        self.stateMachine.request_mode(f"dashboard_{dataDict['Value']}_button")

    def handle_calibration(self, dataDict, socketId):
        self.calibration.handle_calibration_signal(dataDict, socketId)

    def handle_get_current_serial_connection_state(self, socketId):
        self.socketio.emit('current_serial_connection_state', {'data': self.serialConnected}, room=socketId)

    def handle_single_user_session(self, socketId):
        if not self.sessionActive:
            self.sessionActive = True
            self.activeUser = socketId
            self.socketio.emit('session_access', {'data': True}, room=socketId)
            self.send_message_to_brain("RequestSteerLimits", {"Value": True})
        elif self.activeUser == socketId:
            self.socketio.emit('session_access', {'data': True}, room=socketId)
            self.send_message_to_brain("RequestSteerLimits", {"Value": True})
        else:
            self.socketio.emit('session_access', {'data': False}, room=socketId)

    def handle_session_end(self, socketId):
        if self.sessionActive and self.activeUser == socketId:
            self.sessionActive = False
            self.activeUser = None

    def handle_save_table_state(self, data):
        try:
            dataDict = json.loads(data)
            os.makedirs(os.path.dirname(self.table_state_file), exist_ok=True)
            with open(self.table_state_file, 'w') as json_file:
                json.dump(dataDict, json_file, indent=4)
            self.socketio.emit('response', {'data': 'Table state saved successfully'})
        except Exception as e:
            self.logger.error(f"Failed to save table state: {e}")

    def handle_load_table_state(self, data):
        try:
            with open(self.table_state_file, 'r') as json_file:
                dataDict = json.load(json_file)
            self.socketio.emit('loadBack', {'data': dataDict})
        except Exception as e:
            self.logger.error(f"Failed to load table state: {e}")

    def update_hardware_data(self):
        while self.running:
            try:
                self.cpuCoreUsage = psutil.cpu_percent(interval=None, percpu=False)
                self.memoryUsage = psutil.virtual_memory().percent
                self.cpuTemperature = round(psutil.sensors_temperatures()['cpu_thermal'][0].current)
            except Exception:
                pass
            time.sleep(1)

    def send_heartbeat(self):
        while self.running:
            try:
                if not self.heartbeat_received and self.sessionActive:
                    self.heartbeat_retries += 1
                    if self.heartbeat_retries < self.heartbeat_max_retries:
                        self.socketio.emit('heartbeat', {'data': 'Heartbeat'})
                    else:
                        self.socketio.emit('heartbeat_disconnect', {'data': 'Heartbeat timeout'})
                        self.sessionActive = False
                        self.activeUser = None
                        self.heartbeat_retries = 0
                    time.sleep(self.heartbeat_time_between_retries)
                else:
                    self.heartbeat_received = False
                    time.sleep(self.heartbeat_time_between_heartbeats)
            except Exception:
                time.sleep(5)

    def send_continuous_messages(self):
        while self.running:
            try:
                for msg, subscriber in self.messages.items():
                    resp = subscriber["obj"].receive()
                    if resp is not None:
                        if msg == "SerialConnectionState":
                            self.serialConnected = resp
                        self.socketio.emit(msg, {"value": resp})
            except Exception:
                pass
            time.sleep(0.1)

    def send_hardware_data_to_frontend(self):
        while self.running:
            try:
                self.socketio.emit('memory_channel', {'data': self.memoryUsage})
                self.socketio.emit('cpu_channel', {
                    'data': {'usage': self.cpuCoreUsage, 'temp': self.cpuTemperature}
                })
            except Exception:
                pass
            time.sleep(1.0)

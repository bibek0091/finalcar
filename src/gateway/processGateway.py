import sys
import time
import logging
import threading
from src.templates.workerprocess import WorkerProcess

try:
    from flask import Flask
    from flask_socketio import SocketIO
except ImportError:
    Flask, SocketIO = None, None

class processGateway(WorkerProcess):
    """
    This process handles the data distribution to the Remote Web Dashboard.
    It spins up a Flask-SocketIO server and broadcasts `DashboardTelemetry` to remote clients.
    """
    def __init__(self, queueList, logger, ready_event=None, debugging=False, port=5005):
        self.logger = logger
        self.debugging = debugging
        self.port = port
        self.socketio = None
        self.server_thread = None
        super(processGateway, self).__init__(queueList, ready_event)

    def _run_socketio(self):
        if Flask is None or SocketIO is None:
            self.logger.error("Flask-SocketIO is NOT installed! Unabled to start Gateway WebSockets.")
            return

        app = Flask(__name__)
        # Enable CORS so the Angular frontend running on port 8000 can connect to port 5005
        self.socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Web Dashboard Client Connected to Telemetry Gateway.")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Web Dashboard Client Disconnected.")

        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        self.logger.info(f"SocketIO Telemetry Gateway started on port {self.port}!")
        self.socketio.run(app, host="0.0.0.0", port=self.port, allow_unsafe_werkzeug=True)

    def _init_threads(self):
        pass

    def run(self):
        self.server_thread = threading.Thread(target=self._run_socketio, daemon=True)
        self.server_thread.start()
        super(processGateway, self).run()

    def process_work(self):
        """
        Poll the WebTelemetry queue and broadcast it to WebSockets.
        """
        telemetry_queue = self.queuesList.get("WebTelemetry")

        if telemetry_queue is None or self.socketio is None:
            time.sleep(1)
            return

        try:
            if not telemetry_queue.empty():
                payload = telemetry_queue.get_nowait()
                # Broadcast non-blocking JSON to all connected Web Dashboard clients
                self.socketio.emit("telemetry", payload)
        except Exception:
            pass

        # Small delay to prevent 100% CPU lock while polling
        time.sleep(0.01)

    def stop(self):
        self.logger.info("Shutting down SocketIO Telemetry Gateway...")
        super(processGateway, self).stop()


import os
import time
import threading
import http.server
import socketserver
from src.templates.workerprocess import WorkerProcess

class processDashboard(WorkerProcess):
    """
    This process serves the Angular 18 Dashboard UI.
    It spins up a lightweight background HTTP server to serve the static frontend files.
    """
    def __init__(self, queueList, logging, port=8000):
        super(processDashboard, self).__init__(queueList)
        self.logger = logging
        self.port = port
        self.httpd = None
        self.server_thread = None

    def _run_server(self):
        """Runs the HTTP server in a separate thread"""
        base_dir = os.path.dirname(__file__)
        
        # Angular 18 typically builds to frontend/dist/dashboard/browser
        # We will check multiple possible paths to find the index.html
        possible_paths = [
            os.path.join(base_dir, "frontend", "dist", "dashboard", "browser"),
            os.path.join(base_dir, "frontend", "dist", "dashboard"),
            os.path.join(base_dir, "frontend", "dist"),
            os.path.join(base_dir, "frontend")
        ]
        
        web_dir = None
        for p in possible_paths:
            if os.path.exists(os.path.join(p, "index.html")):
                web_dir = p
                break
        
        if web_dir is None:
            self.logger.warning("Could not find Angular index.html. Did you run 'npm run build' inside src/dashboard/frontend?")
            web_dir = os.path.join(base_dir, "frontend") # Fallback
        
        # Change working directory to the frontend folder
        os.chdir(web_dir)
        
        # Silence the standard HTTP server logs to keep your terminal clean
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass 

        try:
            # Prevent "Address already in use" errors
            socketserver.TCPServer.allow_reuse_address = True
            self.httpd = socketserver.TCPServer(("", self.port), QuietHandler)
            self.logger.info(f"Dashboard UI Server started! Open http://localhost:{self.port} or http://<raspberry-pi-ip>:{self.port} in your browser.")
            self.httpd.serve_forever()
        except Exception as e:
            self.logger.error(f"Dashboard server failed to start: {e}")

    def run(self):
        """ Override run to start the server thread before the main loop """
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        super(processDashboard, self).run()

    def process_work(self):
        """ 
        The actual work loop. 
        Since the Gateway handles the WebSockets, this process just needs to keep the HTTP server alive.
        """
        time.sleep(0.5)

    def stop(self):
        """ Ensure the web server shuts down gracefully when main.py is stopped """
        if self.httpd:
            self.logger.info("Shutting down Dashboard UI server...")
            self.httpd.shutdown()
            self.httpd.server_close()
        super(processDashboard, self).stop()
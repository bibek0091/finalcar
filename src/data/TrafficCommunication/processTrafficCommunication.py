import sys
sys.path.insert(0, '../../')

from src.templates.workerprocess import WorkerProcess
from src.data.TrafficCommunication.threads.threadTrafficCommunication import threadTrafficCommunication

class processTrafficCommunication(WorkerProcess):
    """
    TrafficCommunication Process
    Manages the thread for receiving UDP position data and TCP traffic communication,
    writing findings to shared memory.
    """
    def __init__(self, queueList, logging, car_id, ready_event=None, debugging=False):
        super(processTrafficCommunication, self).__init__(queueList, ready_event=ready_event)
        self.queuesList = queueList
        self.logging = logging
        self.car_id = car_id
        self.debugging = debugging

    def _init_threads(self):
        """Initializes the TrafficCommunication thread."""
        traffic_thread = threadTrafficCommunication(
            self.queuesList,
            pause=0.01
        )
        self.threads.append(traffic_thread)

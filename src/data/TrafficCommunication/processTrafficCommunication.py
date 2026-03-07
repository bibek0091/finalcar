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
    def __init__(self, queueList):
        super(processTrafficCommunication, self).__init__(queueList)
        self.queuesList = queueList

    def _init_threads(self):
        """Initializes the TrafficCommunication thread."""
        traffic_thread = threadTrafficCommunication(
            self.queuesList,
            pause=0.01
        )
        self.threads.append(traffic_thread)

import sys
sys.path.insert(0, '../../')

from src.templates.workerprocess import WorkerProcess
from src.data.Semaphores.threads.threadSemaphores import threadSemaphores

class processSemaphores(WorkerProcess):
    """
    Semaphores Process
    Manages the thread for listening to UDP 5007 semaphore broadcasts
    and posting the traffic light states to shared memory.
    """
    def __init__(self, queueList):
        super(processSemaphores, self).__init__(queueList)
        self.queuesList = queueList

    def _init_threads(self):
        """Initializes the Semaphores listening thread."""
        semaphore_thread = threadSemaphores(
            self.queuesList,
            pause=0.2  # 5 Hz = 0.2s pause
        )
        self.threads.append(semaphore_thread)

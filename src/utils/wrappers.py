import numpy as np
from collections import deque
from src.utils import config

class FrameStacker:
    """Stacks the last N frames to provide temporal context to the CNN."""
    def __init__(self, stack_size=config.STACK_SIZE):
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)

    def reset(self, first_frame):
        for _ in range(self.stack_size):
            self.frames.append(first_frame)
        return self.get_stack()

    def append(self, frame):
        self.frames.append(frame)
        return self.get_stack()

    def get_stack(self):
        # Returns shape (84, 84, 4)
        return np.stack(self.frames, axis=-1)
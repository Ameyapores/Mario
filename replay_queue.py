"""A queue for storing previous experiences to sample from."""
import numpy as np


class ReplayQueue(object):
    """A replay queue for replaying previous experiences."""

    def __init__(self, size: int) -> None:
        """
        Initialize a new replay buffer with a given size.

        Args:
            size: the size of the replay buffer
                  (the number of previous experiences to store)

        Returns:
            None

        """
        # initialize the queue data-structure as a list of nil values
        self.queue = [None] * size
        # setup variables for the index and top
        self.index = 0
        self.top = 0

    def __repr__(self) -> str:
        """Return an executable string representation of self."""
        return '{}(size={})'.format(self.__class__.__name__, self.size)

    @property
    def size(self) -> int:
        """Return the size of the queue."""
        return len(self.queue)

    def push(self,
        s: np.ndarray,
        a: int,
        r: int,
    ) -> None:
        """
        Push a new experience onto the queue.

        Args:
            s: the current state
            a: the action to get from current state `s` to next state `s2`
            r: the reward resulting from taking action `a` in state `s`
        Returns:
            None

        """
        # push the variables onto the queue
        self.queue[self.index] = s, a, r
        # increment the index
        self.index = (self.index + 1) % self.size
        # increment the top pointer
        if self.top < self.size:
            self.top += 1

# explicitly define the outward facing API of this module
__all__ = [ReplayQueue.__name__]

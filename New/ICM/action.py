
import numpy as np

#THE FIRST ACTION MUST BE NOOP\n",
ACTIONS = [[0, 0, 0, 0, 0, 0, 0], #0 - no button,\n",
           [1, 0, 0, 0, 0, 0, 0], #1 - up only (to climb vine)\n",
           [0, 0, 1, 0, 0, 0, 0], #2 - left only\n",
           [0, 1, 0, 0, 0, 0, 0], #3 - down only (duck, down pipe)\n",
           [0, 0, 0, 1, 0, 0, 0], #4 - right only\n",
           [0, 0, 0, 0, 0, 1, 0], #5 - run only\n",
           [0, 0, 0, 0, 1, 0, 0]] #6 - jump only\n",

ACTIONS = np.array(ACTIONS)
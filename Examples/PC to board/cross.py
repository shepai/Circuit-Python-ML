"""
Training on device can be more trouble than its worth.

Training on a computer and then corssing the weights over is more ideal.
This tutorial uses an sd card to store weights previously trained, and sets the network weights to this.

"""

from CPML import *
import ulab.numpy as np


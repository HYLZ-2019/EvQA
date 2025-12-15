import sys
import os
# This is eventvl/sparse_e2vid/__init__.py. I want eventvl/sparse_e2vid to be the root of import temporarily.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .video_reconstructor import VideoReconstructor, VideoReconstructor_Adaptive
sys.path.pop()
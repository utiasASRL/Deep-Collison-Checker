print("Running .startup.py")
import sys
import os
user = os.environ['USER']
sys.path.insert(0, "/home/" + user + "/catkin_ws/")
from src.dashboard import *
print("imported dashboard")
d = Dashboard()
print("d = Dashboard()")

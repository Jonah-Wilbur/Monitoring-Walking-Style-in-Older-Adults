import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Gyroscope_data import RawGyroscopeXYZData
from Accelerometer_data import RawAccelerometerXYZData
def decision_tree(window):
    if z_gyro_fft_peak > 100: # if walking
        if numer_of_FFT_peaks == 1: # 
            return 'normal_walking'
        elif yaw_variance_FFT_disorganized == True:
            return 'up/down_stairs'
    elif moving == False and in_bathroom == True:
        if yaw_variance * roll_variance < cutoff:
            return 'brushing_teeth'
        elif yaw_variance * roll_variance >= washing_face_cutoff and acceleration > washing_face_threshold:
            return 'washing_face'
        else:
            return 'other_bathroom_activity'
    else:
        if eating==True:
            return 'eating'
        elif motion==0:
            return 'sitting'
        else:
            return 'other_activity'
    
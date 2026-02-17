import numpy as np

def detect_gas(chunk):

    chunk['slope1'] = np.gradient(chunk['MOX1(Ohms)'], chunk['Time(sec)'])
    chunk['slope3'] = np.gradient(chunk['MOX3(Ohms)'], chunk['Time(sec)'])
    chunk['slope4'] = np.gradient(chunk['MOX4(Ohms)'], chunk['Time(sec)'])

    Min_slope1 = chunk['slope1'].min()
    Min_slope3 = chunk['slope3'].min()

    start_time = chunk['Time(sec)'].iloc[0]

    min_time_mox1 = chunk.loc[chunk['MOX1(Ohms)'].idxmin(), 'Time(sec)']
    max_time_mox1 = chunk.loc[chunk['MOX1(Ohms)'].idxmax(), 'Time(sec)']

    time_diff1 = abs(min_time_mox1 - start_time)
    time_diff2 = abs(max_time_mox1 - min_time_mox1)

    # MOX1 detection
    if time_diff2 <= 39:

        if 1.5 <= time_diff1 <= 11:
            return "NO2 / SO2 / NH3 detected"

        if 4 < time_diff1 < 7 and -450 < Min_slope1 < -200:
            return "NO2 5–7 ppm"

        elif 7 < time_diff1 < 12 and Min_slope1 < -450:
            return "NO2 10 ppm"

    # MOX3 detection
    if Min_slope3 < -500:
        return "CH4 detected"

    return "No gas detected"

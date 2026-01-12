import numpy as np
import scipy.signal as signal

import numpy as np

import numpy as np

def r_peak_detection(ecg_signal, K=20, L=50):
    """
    R-Wave Localization based on sliding window local-mean deviation.

    Parameters
    ----------
    ecg_signal : array-like
        ECG signal samples.
    K : int
        Half-width of the sliding window.
    L : int
        Minimum distance between consecutive R-wave candidates.

    Returns
    -------
    points : list
        Indices of detected R-wave positions.
    """

    ecg_signal = np.asarray(ecg_signal)
    n = len(ecg_signal)

    points = []          # stores detected R-wave positions
    B1_prev = None       # previous significant point
    dev_prev = None      # deviation at previous B1

    # Step 2: slide index i from K to n-K
    for i in range(K, n - K):

        # Step 3: local window and mean
        window = ecg_signal[i - K : i + K + 1]
        local_mean = np.mean(window)

        max_deviation = 0
        B1_candidate = i

        # Step 4: find maximum deviation in window
        for j in range(i - K, i + K + 1):
            deviation = abs(ecg_signal[j] - local_mean)
            if deviation > max_deviation:
                max_deviation = deviation
                B1_candidate = j

        # Step 5: evaluate and update significant points
        if B1_prev is None:
            # first detected point
            B1_prev = B1_candidate
            dev_prev = max_deviation
            points.append(B1_prev)

        else:
            distance = abs(B1_candidate - B1_prev)

            if distance < L:
                # keep the point with higher deviation
                if max_deviation > dev_prev:
                    B1_prev = B1_candidate
                    dev_prev = max_deviation
                    points[-1] = B1_prev
            else:
                # accept new heartbeat
                B1_prev = B1_candidate
                dev_prev = max_deviation
                points.append(B1_prev)

    # Step 6: return detected R-wave positions
    return points
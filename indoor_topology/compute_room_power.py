import numpy as np
from scipy.io import loadmat


def load_room_power(mat_path):
    """Load room power data from a MATLAB `.mat` file.

    The MAT file should contain arrays:
        - `frequency`: array of shape `(n_freq,)`
        - `room_power`: array of shape `(n_room, n_freq)`
        - optional `room_ids`: array mapping rows of `room_power` to room id.
          If missing, room IDs are assumed to start from 1.
    """
    data = loadmat(mat_path)
    freqs = np.squeeze(data.get('frequency'))
    power = np.asarray(data.get('room_power'))
    if power.ndim == 1:
        power = power[:, None]
    room_ids = np.squeeze(data.get('room_ids'))
    if room_ids is None or room_ids.size == 0:
        room_ids = np.arange(power.shape[0]) + 1

    power_dict = {}
    for i, f in enumerate(freqs):
        freq_power = {}
        for j, rid in enumerate(room_ids):
            freq_power[int(rid)] = float(power[j, i])
        power_dict[float(f)] = freq_power

    return freqs.astype(float), power_dict

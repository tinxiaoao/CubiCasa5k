import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from .extract_rooms import extract_rooms
from .compute_room_power import load_room_power


def _power_map(region_id_map, room_power):
    """Return a 2-D array with power values placed according to region ids."""
    heat = np.full(region_id_map.shape, np.nan, dtype=float)
    for rid, p in room_power.items():
        heat[region_id_map == rid] = p
    return heat


def visualize_room_power(wall_svg_path: str, rough_img_path: str, mat_path: str):
    """Visualize room reception power on the roughcast image.

    Parameters
    ----------
    wall_svg_path : str
        Path to ``wall_svg.png`` image.
    rough_img_path : str
        Path to the roughcast image (``svgImg_roughcast.png``).
    mat_path : str
        Path to MATLAB ``.mat`` file containing power results.
    """
    wall_img = Image.open(wall_svg_path).convert('P')
    wall_array = np.array(wall_img)
    region_id_map, rooms = extract_rooms(wall_array)

    freqs, power_data = load_room_power(mat_path)
    rough_img = np.array(Image.open(rough_img_path).convert('RGBA'))

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(rough_img)
    init_power = _power_map(region_id_map, power_data[freqs[0]])
    hm = ax.imshow(init_power, cmap='jet', alpha=0.6)
    cbar = plt.colorbar(hm, ax=ax)
    cbar.set_label('Received power (dBm)')

    axfreq = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(axfreq, 'Frequency', float(freqs[0]), float(freqs[-1]),
                    valinit=float(freqs[0]), valstep=np.diff(freqs).min())

    def update(val):
        freq = slider.val
        room_power = power_data.get(freq, power_data[freqs[0]])
        hm.set_data(_power_map(region_id_map, room_power))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    # Plot per room power curves
    plt.figure()
    for room in rooms:
        rid = room['id']
        y = [power_data[f].get(rid, np.nan) for f in freqs]
        plt.plot(freqs, y, label=f'Room {rid}')
    plt.xlabel('Frequency')
    plt.ylabel('Received power (dBm)')
    plt.legend()
    plt.show()

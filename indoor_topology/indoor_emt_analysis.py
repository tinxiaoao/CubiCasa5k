# indoor_emt_analysis.py
# Functions: MATLAB integration, power-frequency curves, optimized interactive heatmap
# Dependencies: matlab.engine, numpy, matplotlib, pillow, ipywidgets, scipy

import numpy as np
import matlab.engine
from matplotlib import pyplot as plt
from PIL import Image
from ipywidgets import interact, SelectionSlider
from indoor_topology.extract_rooms import extract_rooms
from scipy.ndimage import center_of_mass


# Fetch results from MATLAB
def fetch_matlab_results(script_path):
    print("Running MATLAB script in the background...")
    eng = matlab.engine.start_matlab()
    eng.eval(f"run('{script_path}')", nargout=0)
    room_ids = np.array(eng.eval("resultTable{:,1}")).flatten().astype(int)
    E_matrix = np.vstack([np.array(p).flatten() for p in eng.eval("resultTable{:,2}")])
    freq_vector = np.array(eng.workspace['frequency']).flatten()
    eng.quit()
    print("MATLAB calculation completed.")
    return room_ids, E_matrix, freq_vector


# Plot E-frequency curves
def plot_E_curves(room_ids, E_matrix, freq_vector):
    plt.figure()
    for i, room_id in enumerate(room_ids):
        plt.plot(freq_vector, E_matrix[i, :], label=f"Room {room_id}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Received E-filed (V/m)")
    plt.title("Received E-field vs Frequency")
    plt.legend()
    plt.grid()
    plt.show()


# Optimized interactive heatmap visualization with room IDs and power values displayed
def interactive_heatmap(region_id_map, room_ids, E_matrix, freq_vector, base_image_path):
    base_img = Image.open(base_image_path).convert("RGBA")
    cmap = plt.cm.jet
    E_min, E_max = E_matrix.min(), E_matrix.max()

    freq_options = {f"{freq / 1e9:.2f} GHz": idx for idx, freq in enumerate(freq_vector)}

    # Compute centers for room IDs display
    room_centers = {}
    for room_id in room_ids:
        mask = region_id_map == room_id
        room_centers[room_id] = center_of_mass(mask)

    heatmap_images = []
    for freq_idx in range(len(freq_vector)):
        overlay = np.zeros(region_id_map.shape + (4,), dtype=np.uint8)
        for i, room_id in enumerate(room_ids):
            norm_E = (E_matrix[i, freq_idx] - E_min) / (E_max - E_min)
            color = cmap(norm_E, alpha=0.6)
            overlay[region_id_map == room_id] = (np.array(color[:4]) * 255).astype(np.uint8)

        overlay_img = Image.fromarray(overlay, mode='RGBA')
        composite_img = Image.alpha_composite(base_img, overlay_img)
        heatmap_images.append(composite_img)

    def update(freq_label):
        freq_idx = freq_options[freq_label]
        fig, ax = plt.subplots(figsize=(10, 8))
        img_display = ax.imshow(heatmap_images[freq_idx])
        ax.set_title(f"Heatmap at Frequency {freq_label}")
        ax.axis("off")

        for i, room_id in enumerate(room_ids):
            y, x = room_centers[room_id]
            ax.text(x, y, f"Room {room_id}\n{E_matrix[i, freq_idx]:.2f}", color='white', fontsize=8,
                    ha='center', va='center', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=E_min, vmax=E_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Received E-filed (V/m)")
        cbar.ax.set_aspect('auto')

        plt.show()

    interact(
        update,
        freq_label=SelectionSlider(
            options=list(freq_options.keys()),
            description='Frequency',
            continuous_update=False,
            style={'description_width': 'initial'}
        )
    )


# Main execution block
if __name__ == "__main__":
    matlab_script = r"E:\code\IndoorEMT\debug_roomself_wall.m"

    # 注意更改此文件夹中的图片，以便于绘图查看
    wall_svg_path = "wall_svg.png"
    rough_img_path = "svgImg_roughcast.png"

    wall_label_img = Image.open(wall_svg_path).convert('P')
    wall_label_array = np.array(wall_label_img)

    room_ids, E_matrix, freq_vector = fetch_matlab_results(matlab_script)

    plot_E_curves(room_ids, E_matrix, freq_vector)

    region_id_map, rooms, _, _ = extract_rooms(wall_label_array)

    interactive_heatmap(region_id_map, room_ids, E_matrix, freq_vector, rough_img_path)

# jupyter notebook
# import sys
# sys.path.append(r"E:\code\CubiCasa5k")  # 确认是实际项目路径

# indoor_emt_analysis.py
# Functions: MATLAB integration, power-frequency curves, interactive heatmap
# Dependencies: matlab.engine, numpy, matplotlib, pillow, ipywidgets

import numpy as np
import matlab.engine
from matplotlib import pyplot as plt
from PIL import Image
from ipywidgets import interact, IntSlider
from indoor_topology.extract_rooms import extract_rooms

# Fetch results from MATLAB
def fetch_matlab_results(script_path):
    print("Running MATLAB script in the background...")
    eng = matlab.engine.start_matlab()
    eng.eval(f"run('{script_path}')", nargout=0)
    room_ids = np.array(eng.eval("resultTable{:,1}")).flatten().astype(int)
    power_matrix = np.vstack([np.array(p).flatten() for p in eng.eval("resultTable{:,2}")])
    freq_vector = np.array(eng.workspace['frequency']).flatten()
    eng.quit()
    print("MATLAB calculation completed.")
    return room_ids, power_matrix, freq_vector

# Plot power-frequency curves
def plot_power_curves(room_ids, power_matrix, freq_vector):
    plt.figure()
    for i, room_id in enumerate(room_ids):
        plt.plot(freq_vector, power_matrix[i, :], label=f"Room {room_id}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Received Power (dBm)")
    plt.title("Received Power vs Frequency")
    plt.legend()
    plt.grid()
    plt.show()

# Interactive heatmap visualization
def interactive_heatmap(region_id_map, room_ids, power_matrix, freq_vector, base_image_path):
    def update(freq_idx):
        base_img = Image.open(base_image_path).convert("RGBA")
        overlay = np.zeros(region_id_map.shape+(4,), dtype=np.uint8)
        cmap = plt.cm.jet
        power_min, power_max = power_matrix.min(), power_matrix.max()

        for i, room_id in enumerate(room_ids):
            norm_power = (power_matrix[i, freq_idx] - power_min) / (power_max - power_min)
            color = cmap(norm_power, alpha=0.6)
            overlay[region_id_map == room_id] = (np.array(color[:4])*255).astype(np.uint8)

        overlay_img = Image.fromarray(overlay, mode='RGBA')
        composite_img = Image.alpha_composite(base_img, overlay_img)
        plt.figure(figsize=(8, 8))
        plt.imshow(composite_img)
        plt.title(f"Heatmap at Frequency {freq_vector[freq_idx]:.2f} Hz")
        plt.axis("off")
        plt.show()

    interact(update, freq_idx=IntSlider(min=0, max=len(freq_vector)-1, step=1, description="Frequency Index"))

# Main execution block
if __name__ == "__main__":
    matlab_script = r"E:\code\IndoorEMT\debug_roomself_wall.m"

    wall_svg_path = "wall_svg.png"
    rough_img_path = "svgImg_roughcast.png"

    wall_label_img = Image.open(wall_svg_path).convert('P')
    wall_label_array = np.array(wall_label_img)

    room_ids, power_matrix, freq_vector = fetch_matlab_results(matlab_script)

    plot_power_curves(room_ids, power_matrix, freq_vector)

    region_id_map, rooms, _, _ = extract_rooms(wall_label_array)

    interactive_heatmap(region_id_map, room_ids, power_matrix, freq_vector, rough_img_path)

import os
from pathlib import Path
import numpy as np
import matlab.engine
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from ipywidgets import widgets, interactive, Output, VBox
from IPython.display import display, clear_output
from indoor_topology.extract_rooms import extract_rooms
import time


def fetch_matlab_results(script_path):
    print("Running MATLAB script ...")
    eng = matlab.engine.start_matlab()
    eng.eval(f"run('{script_path}')", nargout=0)
    resultMatrix_E = np.asarray(eng.workspace['resultMatrix_E'])
    resultMatrix_P = np.asarray(eng.workspace['resultMatrix_P'])
    freq_vector = np.asarray(eng.workspace['frequency']).flatten()
    S_in_dBm = np.asarray(eng.workspace['S_in_dBm']).flatten()
    eng.quit()
    room_ids = resultMatrix_E[:, 0].astype(int)
    E_matrix = resultMatrix_E[:, 1:]
    P_matrix = resultMatrix_P[:, 1:]
    source_rooms = room_ids[np.isfinite(S_in_dBm) & (S_in_dBm > -np.inf)]
    source_powers = {r: p for r, p in zip(room_ids, S_in_dBm) if np.isfinite(p) and p > -np.inf}
    return room_ids, E_matrix, P_matrix, freq_vector, source_rooms, source_powers


def plot_and_save_curves(room_ids, matrix, freq_vector, source_rooms,
                         ylabel, filename):
    plt.figure(figsize=(7.16, 4.0))
    for i, rid in enumerate(room_ids):
        label = f"Room_{rid}" + (" (Tx)" if rid in source_rooms else "")
        plt.plot(freq_vector / 1e9, matrix[i], label=label, linewidth=1)

    plt.xlabel("Frequency (GHz)", fontsize=9, fontname='Times New Roman')
    plt.ylabel(ylabel, fontsize=9, fontname='Times New Roman')

    if "E-field" in ylabel:
        plt.title("Variation of Received E-field Across Rooms with Frequency",
                  fontsize=10, fontname='Times New Roman', fontweight='normal')
    else:
        plt.title("Variation of Received Power Across Rooms with Frequency",
                  fontsize=10, fontname='Times New Roman', fontweight='normal')

    # 图例放置到底部
    plt.legend(loc='upper center', fontsize=8, prop={'family': 'Times New Roman'},
               bbox_to_anchor=(0.5, -0.15), ncol=6, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(filename, dpi=600, format='png', bbox_inches='tight')
    plt.close()


def save_power_to_txt( P_matrix, freq_vector, freqs_ghz, filename):
    target_freqs_hz = np.array(freqs_ghz) * 1e9
    idxs = []
    for f in target_freqs_hz:
        matches = np.where(np.isclose(freq_vector, f, atol=1))[0]
        if len(matches) == 0:
            raise ValueError(f"Requested frequency {f / 1e9:.3f} GHz not found in freq_vector.")
        idxs.append(matches[0])
    with open(filename, 'w') as f:
        for fghz, idx in zip(freqs_ghz, idxs):
            f.write(f"Frequency: {fghz:.2f} GHz\nPower(dBm)\n")
            for p in P_matrix[:, idx]:
                f.write(f"{p:.2f}\n")
            f.write('\n')


def make_overlay(region_id_map, room_ids, values, norm, cmap, alpha=0.7):
    overlay = np.zeros(region_id_map.shape + (4,), dtype=np.uint8)
    for rid, val in zip(room_ids, values):
        nv = norm(val)
        r, g, b, _ = cmap(nv, bytes=True)
        overlay[region_id_map == rid] = (r, g, b, int(alpha * 255))
    return overlay


def draw_single_heatmap(ax, base_img_rgba, overlay_rgba,
                        room_centers, room_ids, values,
                        norm, cmap,
                        source_rooms, source_powers,
                        title_text, value_unit):
    composite = Image.alpha_composite(base_img_rgba, Image.fromarray(overlay_rgba))
    ax.imshow(composite, aspect='equal')
    ax.set_title(title_text, fontsize=10, fontweight='normal', fontname='Times New Roman')
    ax.axis('off')

    labeled_positions = []

    for rid, val in zip(room_ids, values):
        y, x = room_centers[rid]

        # 获取背景颜色以计算文字颜色
        normalized_value = norm(val)
        background_color = cmap(normalized_value)
        txt_color = get_text_color(background_color)

        # 计算房间面积，动态调整字体大小
        room_area = np.sum(region_id_map == rid)
        fontsize = calculate_fontsize(room_area)

        room_label = f"({rid})\n{val:.1f} {value_unit}"

        # 避免文字重叠
        offset_y = 0
        for prev_x, prev_y in labeled_positions:
            if np.sqrt((prev_x - x) ** 2 + (prev_y - y) ** 2) < 20:
                offset_y += 20

        ax.text(x, y + offset_y, room_label,
                color=txt_color, ha='center', va='center',
                fontsize=fontsize, fontweight='normal', fontname='Times New Roman',
                bbox=dict(facecolor='none', edgecolor='none', pad=1),
                zorder=3)

        labeled_positions.append((x, y + offset_y))

        if rid in source_rooms:
            ax.add_patch(plt.Circle((x, y), radius=10, edgecolor='red', facecolor='none', linewidth=1.5, zorder=4))
            ax.text(x, y - 35, f"Source: {source_powers[rid]:.1f} {value_unit}",
                    color='red', ha='center', va='center',
                    fontsize=fontsize - 1, fontweight='normal', fontname='Times New Roman',
                    zorder=5)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"{title_text.split(' at ')[0]} ({value_unit})",
                   fontsize=8, fontname='Times New Roman')


# 辅助函数
def calculate_fontsize(area, min_size=3, max_size=7):
    return np.clip(area * 0.0005, min_size, max_size)


def get_text_color(background_rgb):
    r, g, b = background_rgb[:3]
    brightness = (0.299 * r + 0.587 * g + 0.114 * b)
    return 'white' if brightness < 0.5 else 'black'


def interactive_heatmap(region_id_map, room_ids, matrix, freq_vector,
                        base_img_path, source_rooms, source_powers,
                        label, value_unit="dBm", vmin=None, vmax=None):
    base_img = Image.open(base_img_path).convert("RGBA")
    base_img_rgba = base_img
    room_centers = {rid: center_of_mass(region_id_map == rid) for rid in room_ids}

    vmin = matrix.min() if vmin is None else vmin
    vmax = matrix.max() if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap('jet')

    dpi_val = 600

    # 明确使用实际频率点 (GHz)
    freq_list_ghz = np.round(freq_vector / 1e9, 2)

    freq_slider = widgets.SelectionSlider(
        options=freq_list_ghz,
        value=freq_list_ghz[0],
        description='Frequency (GHz):',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    freq_dropdown = widgets.Dropdown(
        options=freq_list_ghz,
        value=freq_list_ghz[0],
        description='Input (GHz):',
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    # 使用observe实现明确双向同步
    def slider_to_dropdown(change):
        freq_dropdown.value = change.new

    def dropdown_to_slider(change):
        freq_slider.value = change.new

    freq_slider.observe(slider_to_dropdown, names='value')
    freq_dropdown.observe(dropdown_to_slider, names='value')

    output = Output()

    def update_plot(freq_ghz):
        matches = np.where(np.isclose(freq_vector, freq_ghz * 1e9, atol=1))[0]
        if len(matches) == 0:
            with output:
                clear_output(wait=True)
                print(f"No exact frequency match found for {freq_ghz:.2f} GHz.")
            return
        fi = matches[0]

        values = matrix[:, fi]
        overlay_rgba = make_overlay(region_id_map, room_ids, values, norm, cmap)

        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(7.16, 5), dpi=600)
            title_text = f"{label} Distribution at {freq_ghz:.2f} GHz"
            draw_single_heatmap(ax, base_img_rgba, overlay_rgba,
                                room_centers, room_ids, values,
                                norm, cmap,
                                source_rooms, source_powers,
                                title_text=title_text, value_unit=value_unit)
            plt.tight_layout()
            plt.show()

    interactive_plot = interactive(update_plot, freq_ghz=freq_slider)
    display(VBox([interactive_plot, freq_dropdown, output]))


def save_heatmap(region_id_map, room_ids, matrix, freq_vector,
                 base_img_path, target_freqs,
                 filename_prefix,
                 source_rooms, source_powers, value_unit="dBm",
                 vmin=None, vmax=None):
    base_img = Image.open(base_img_path).convert("RGBA")
    base_img_rgba = base_img
    room_centers = {rid: center_of_mass(region_id_map == rid) for rid in room_ids}

    vmin = matrix.min() if vmin is None else vmin
    vmax = matrix.max() if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap('jet')

    for freq in target_freqs:
        matches = np.where(np.isclose(freq_vector, freq * 1e9, atol=1))[0]
        if len(matches) == 0:
            raise ValueError(f"Exact frequency match for {freq:.2f} GHz not found.")
        fi = matches[0]

        values = matrix[:, fi]
        ov = make_overlay(region_id_map, room_ids, values, norm, cmap)

        fig, ax = plt.subplots(figsize=(7.16, 5), dpi=600)

        title = f"Average Received Power Distribution at {freq:.2f} GHz"

        draw_single_heatmap(ax, base_img_rgba, ov,
                            room_centers, room_ids, values,
                            norm, cmap,
                            source_rooms, source_powers,
                            title_text=title, value_unit=value_unit)

        out_path = f"{filename_prefix}_{freq:.2f}GHz.png"
        fig.savefig(out_path, bbox_inches='tight', dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    start_time = time.time()
    output_dir = r"E:\manuscript\journal\2025Tap\calculate_result\CubiCase5K\7795\c_2.4GHz"
    matlab_script = r"E:\code\IndoorEMT\debug_roomself_wall.m"
    wall_svg_path = "wall_svg.png"
    rough_img_path = "svgImg_roughcast.png"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    wall_label_array = np.asarray(Image.open(wall_svg_path).convert('P'))

    (room_ids, E_matrix, P_matrix, freq_vector,
     source_rooms, source_powers) = fetch_matlab_results(matlab_script)

    plot_and_save_curves(room_ids, E_matrix, freq_vector, source_rooms,
                         "Received E-field (V/m)",
                         os.path.join(output_dir, "E_field_curve.png"))
    plot_and_save_curves(room_ids, P_matrix, freq_vector, source_rooms,
                         "Received Power (dBm)",
                         os.path.join(output_dir, "Power_curve.png"))
    region_id_map, *_ = extract_rooms(wall_label_array)

    # 场强 (E-field)
    interactive_heatmap(region_id_map, room_ids, E_matrix, freq_vector,
                        rough_img_path, source_rooms, source_powers,
                        label="Average Received E-field", value_unit="V/m",
                        vmin=E_matrix.min(), vmax=E_matrix.max())

    # 接收功率 (Power)
    interactive_heatmap(region_id_map, room_ids, P_matrix, freq_vector,
                        rough_img_path, source_rooms, source_powers,
                        label="Average Received Power", value_unit="dBm",
                        vmin=P_matrix.min(), vmax=P_matrix.max())

    target_freqs = [2.4, 5, 28]

    save_heatmap(region_id_map, room_ids, P_matrix, freq_vector,
                 rough_img_path,
                 target_freqs,
                 os.path.join(output_dir, "Power_heatmap"),
                 source_rooms, source_powers,
                 value_unit="dBm",
                 vmin=P_matrix.min(), vmax=P_matrix.max())

    save_power_to_txt(P_matrix, freq_vector, target_freqs, os.path.join(output_dir, "frequency_powers.txt"))
    # 运行完毕提示与运行时间计算
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n{'-' * 50}")
    print("🎯 程序运行完毕！")
    print(f"⏱️ 总运行时间: {elapsed_time:.2f} 秒")
    print(f"{'-' * 50}\n")

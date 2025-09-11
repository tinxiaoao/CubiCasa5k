import os
from pathlib import Path
import numpy as np
import matlab.engine
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import center_of_mass
from indoor_topology.extract_rooms import extract_rooms
from IPython.display import display, clear_output
from ipywidgets import widgets, interactive, Output, VBox
import time

# ===== 固定色尺（dBm） =====
GLOBAL_VMIN_DBM = -140.0
GLOBAL_VMAX_DBM = -20.0


# ---------- 小工具：由线性频率向量计算索引 ----------
def freq_to_index(freq_vector_hz: np.ndarray, freq_ghz: float) -> int:
    """freq_vector 为 0.2e9:1e8:67e9 的线性网格；这里用整型索引定位"""
    f0 = float(freq_vector_hz[0])
    df = float(freq_vector_hz[1] - freq_vector_hz[0])
    n = int(freq_vector_hz.size)
    f = float(freq_ghz) * 1e9
    idx = int(round((f - f0) / df))
    if idx < 0 or idx >= n:
        raise ValueError(f"{freq_ghz:.2f} GHz out of range [{f0 / 1e9:.2f}, {freq_vector_hz[-1] / 1e9:.2f}]")
    # 可做一次严格校验（允许极微小误差）
    if abs(freq_vector_hz[idx] - f) > 1.0:  # 1 Hz 容差
        raise ValueError(f"{freq_ghz:.2f} GHz not on the defined grid.")
    return idx


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
    E_matrix = resultMatrix_E[:, 1:]  # 本脚本不再使用 E，但保留返回
    P_matrix = resultMatrix_P[:, 1:]
    source_rooms = room_ids[np.isfinite(S_in_dBm) & (S_in_dBm > -np.inf)]
    source_powers = {r: p for r, p in zip(room_ids, S_in_dBm) if np.isfinite(p) and p > -np.inf}
    return room_ids, E_matrix, P_matrix, freq_vector, source_rooms, source_powers


def save_power_to_txt(P_matrix, freq_vector, freqs_ghz, filename):
    idxs = [freq_to_index(freq_vector, fghz) for fghz in freqs_ghz]
    with open(filename, 'w') as f:
        for fghz, idx in zip(freqs_ghz, idxs):
            f.write(f"Frequency: {fghz:.2f} GHz\nPower(dBm)\n")
            for p in P_matrix[:, idx]:
                f.write(f"{float(p):.2f}\n")
            f.write('\n')


def make_overlay(region_id_map, room_ids, values, norm, cmap, alpha=0.7):
    overlay = np.zeros(region_id_map.shape + (4,), dtype=np.uint8)
    for rid, val in zip(room_ids, values):
        nv = np.clip(norm(float(val)), 0.0, 1.0)
        r, g, b, _ = cmap(nv, bytes=True)
        overlay[region_id_map == rid] = (r, g, b, int(alpha * 255))
    return overlay


def draw_single_heatmap(ax, base_img_rgba, overlay_rgba,
                        region_id_map, room_centers, room_ids, values,
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
        bg = cmap(np.clip(norm(float(val)), 0.0, 1.0))
        txt_color = 'white' if (0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]) < 0.5 else 'black'
        room_area = np.sum(region_id_map == rid)
        fontsize = float(np.clip(room_area * 0.0005, 3, 7))
        lbl = f"({rid})\n{float(val):.1f} {value_unit}"

        offset_y = 0
        for px, py in labeled_positions:
            if np.hypot(px - x, py - y) < 20:
                offset_y += 20
        ax.text(x, y + offset_y, lbl,
                color=txt_color, ha='center', va='center',
                fontsize=fontsize, fontweight='normal', fontname='Times New Roman',
                bbox=dict(facecolor='none', edgecolor='none', pad=1), zorder=3)
        labeled_positions.append((x, y + offset_y))

        if rid in source_rooms:
            ax.add_patch(plt.Circle((x, y), radius=10, edgecolor='red', facecolor='none', linewidth=1.5, zorder=4))
            ax.text(x, y - 35, f"Source: {float(source_powers[rid]):.1f} {value_unit}",
                    color='red', ha='center', va='center',
                    fontsize=max(fontsize - 1, 3), fontweight='normal', fontname='Times New Roman',
                    zorder=5)

    # 固定色尺：−140 ~ −20 dBm
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([norm.vmin, norm.vmax])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    ticks = np.linspace(norm.vmin, norm.vmax, 7)  # [-140,-120,-100,-80,-60,-40,-20]
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"{title_text.split(' at ')[0]} ({value_unit})",
                   fontsize=8, fontname='Times New Roman')


def interactive_power_heatmap(region_id_map, room_ids, matrix, freq_vector,
                              base_img_path, source_rooms, source_powers,
                              value_unit="dBm"):
    """功率交互热力图：固定色尺 [-140, -20] dBm；频点用整数索引"""
    base_img = Image.open(base_img_path).convert("RGBA")
    base_img_rgba = base_img
    room_centers = {rid: center_of_mass(region_id_map == rid) for rid in room_ids}

    norm = plt.Normalize(vmin=GLOBAL_VMIN_DBM, vmax=GLOBAL_VMAX_DBM)
    cmap = plt.get_cmap('jet')

    # 频率选项（GHz），来自线性频率网格
    freq_list_ghz = np.round(freq_vector / 1e9, 2)

    slider = widgets.SelectionSlider(
        options=freq_list_ghz, value=freq_list_ghz[0],
        description='Frequency (GHz):', continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    output = Output()

    def update_plot(freq_ghz):
        # 线性网格 → 用整数索引取频点
        fi = freq_to_index(freq_vector, float(freq_ghz))
        values = matrix[:, fi]
        overlay_rgba = make_overlay(region_id_map, room_ids, values, norm, cmap)

        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(7.16, 5), dpi=600)
            title_text = f"Average Received Power Distribution at {float(freq_ghz):.2f} GHz"
            draw_single_heatmap(ax, base_img_rgba, overlay_rgba,
                                region_id_map, room_centers, room_ids, values,
                                norm, cmap,
                                source_rooms, source_powers,
                                title_text=title_text, value_unit=value_unit)
            plt.tight_layout()
            plt.show()

    display(VBox([interactive(update_plot, freq_ghz=slider), output]))


def save_heatmap(region_id_map, room_ids, matrix, freq_vector,
                 base_img_path, target_freqs,
                 filename_prefix,
                 source_rooms, source_powers,
                 value_unit="dBm"):
    base_img = Image.open(base_img_path).convert("RGBA")
    base_img_rgba = base_img
    room_centers = {rid: center_of_mass(region_id_map == rid) for rid in room_ids}

    # 固定色尺
    norm = plt.Normalize(vmin=GLOBAL_VMIN_DBM, vmax=GLOBAL_VMAX_DBM)
    cmap = plt.get_cmap('jet')

    for freq in target_freqs:
        fi = freq_to_index(freq_vector, freq)
        values = matrix[:, fi]
        ov = make_overlay(region_id_map, room_ids, values, norm, cmap)

        fig, ax = plt.subplots(figsize=(7.16, 5), dpi=600)
        title = f"Average Received Power Distribution at {freq:.2f} GHz"
        draw_single_heatmap(ax, base_img_rgba, ov,
                            region_id_map, room_centers, room_ids, values,
                            norm, cmap,
                            source_rooms, source_powers,
                            title_text=title, value_unit=value_unit)
        out_path = f"{filename_prefix}_{freq:.2f}GHz.png"
        fig.savefig(out_path, bbox_inches='tight', dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    start_time = time.time()
    output_dir = r"E:\manuscript\journal\2025Tap\calculate_result\CubiCase5K\7795\a\room1_dooropen"
    matlab_script = r"E:\code\IndoorEMT\debug_roomself_wall.m"
    wall_svg_path = "wall_svg.png"
    rough_img_path = "svgImg_roughcast.png"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    wall_label_array = np.asarray(Image.open(wall_svg_path).convert('P'))

    (room_ids, _E_matrix, P_matrix, freq_vector,
     source_rooms, source_powers) = fetch_matlab_results(matlab_script)

    # 仅功率曲线（如不需要也可注释掉）
    plt.figure(figsize=(7.16, 4.0))
    for i, rid in enumerate(room_ids):
        label = f"Room_{rid}" + (" (Tx)" if rid in source_rooms else "")
        plt.plot(freq_vector / 1e9, P_matrix[i], label=label, linewidth=1)
    plt.xlabel("Frequency (GHz)", fontsize=9, fontname='Times New Roman')
    plt.ylabel("Received Power (dBm)", fontsize=9, fontname='Times New Roman')
    plt.title("Variation of Received Power Across Rooms with Frequency",
              fontsize=10, fontname='Times New Roman', fontweight='normal')
    plt.legend(loc='upper center', fontsize=8, prop={'family': 'Times New Roman'},
               bbox_to_anchor=(0.5, -0.15), ncol=6, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, "Power_curve.png"),
                dpi=600, format='png', bbox_inches='tight')
    plt.close()

    region_id_map, *_ = extract_rooms(wall_label_array)

    # 固定色尺的静态热力图
    target_freqs = [2.4, 5.0, 28.0]  # 自由修改
    save_heatmap(region_id_map, room_ids, P_matrix, freq_vector,
                 rough_img_path,
                 target_freqs,
                 os.path.join(output_dir, "Power_heatmap"),
                 source_rooms, source_powers,
                 value_unit="dBm")

    # 导出指定频点功率（仅功率列，便于复制）
    save_power_to_txt(P_matrix, freq_vector, target_freqs,
                      os.path.join(output_dir, "frequency_powers.txt"))

    # 交互功率热力图（滑块），固定色尺 [-140, -20] dBm
    interactive_power_heatmap(region_id_map, room_ids, P_matrix, freq_vector,
                              rough_img_path, source_rooms, source_powers,
                              value_unit="dBm")

    print("\n" + "-" * 50)
    print("🎯 程序运行完毕！  固定色尺: [-140, -20] dBm")
    print(f"⏱️ 总运行时间: {time.time() - start_time:.2f} 秒")
    print("-" * 50 + "\n")

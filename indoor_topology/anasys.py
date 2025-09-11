import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# 指定路径
folder_path = r'E:\manuscript\journal\2025Tap\simulate_model\wirelessinsite\7795\simulate_results_test\a_28GHz_660_doorclose'
output_folder = os.path.join(folder_path, '3dB_anasys')

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 房间预测功率值
predicted_values = {
    1: -49.77, 2: -81.74, 3: -112.05, 4: -107.65, 5: -93.74, 6: -124.64,
    7: -102.27, 8: -78.01, 9: -101.55, 10: -73.39, 11: -103.55, 12: -73.3,
    13: -105.51, 14: -140.39, 15: -110.31, 16: -117.93, 17: -98.46, 18: -77.59
}

# 初始化数据结构
room_data = {}

# 遍历文件
for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
    df = pd.read_csv(csv_file, skiprows=2, header=None)
    rx_pts = df.iloc[:, 2]
    power_values = df.iloc[:, 3]

    with open(csv_file, 'r') as file:
        first_line = file.readline()
        room_match = re.search(r'Room_(\d+) --> Room_(\d+)_(\d+\.\d+)m', first_line)
        if room_match:
            room_number = int(room_match.group(2))
            height = room_match.group(3)
        else:
            continue

    target_power = predicted_values.get(room_number)
    if target_power is None:
        continue

    within_3db_pts = rx_pts[np.abs(power_values - target_power) <= 3].tolist()

    if room_number not in room_data:
        room_data[room_number] = {}

    room_data[room_number][height] = {
        'power_values': power_values,
        'within_3db_pts': within_3db_pts
    }

# 分析和绘制每个房间的数据
for room_number, data in room_data.items():
    plt.figure(figsize=(10, 6))

    points_info = []
    for height, details in data.items():
        plt.hist(details['power_values'], bins=20, alpha=0.6, density=True, label=f'{height}m')
        points_info.append(f"高度 {height}m：\n与预测值相差3 dB以内的点数：{len(details['within_3db_pts'])}\n编号：{details['within_3db_pts'] or '无'}")

    # 添加预测值和±3 dB范围
    predicted_power = predicted_values[room_number]
    plt.axvline(predicted_power, color='r', linestyle='--', linewidth=2, label='Predicted Value')
    plt.axvspan(predicted_power - 3, predicted_power + 3, color='green', alpha=0.3, label='±3 dB Range')

    plt.xlabel('Received Power (dBm)')
    plt.ylabel('Probability Density')
    plt.title(f'Power Distribution for Room {room_number}')
    plt.legend()
    plt.grid(alpha=0.5)

    # 保存每个房间的图
    output_filepath = os.path.join(output_folder, f'Room_{room_number}_Power_Distribution.png')
    plt.savefig(output_filepath, dpi=300)
    plt.close()

    # 比较高度点编号一致性
    heights = list(data.keys())
    if len(heights) == 2:
        set1 = set(data[heights[0]]['within_3db_pts'])
        set2 = set(data[heights[1]]['within_3db_pts'])
        common_pts = set1.intersection(set2)
        consistency_info = "一致" if common_pts else "不一致"
    else:
        consistency_info = "只有一个高度的数据"

    # 打印详细分析结果
    print(f"Room {room_number}分析结果:\n")
    print("（1）相差3 dB以内的点数量与编号：")
    for info in points_info:
        print(info)
    print("\n（2）两高度点编号一致性：")
    print(consistency_info)
    print(f"\n功率分布图已保存至：{output_filepath}\n")

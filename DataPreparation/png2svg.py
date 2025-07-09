import cv2, numpy as np, svgwrite
from shapely.geometry import LineString, MultiLineString, GeometryCollection
from shapely.ops import linemerge, unary_union
from pathlib import Path

# ---------- 参数可根据图纸情况调整 ----------
THRESHOLD = 200       # 二值化阈值
MIN_LINE_LEN = 40     # Hough 最短线长
MAX_LINE_GAP = 10     # Hough 最大断裂
CLOSE_KERNEL = (5, 5) # 闭运算 kernel
# -------------------------------------------

def flatten_lines(geom):
    """递归展开 GeometryCollection / MultiLineString，返回 List[LineString]"""
    if geom is None:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, (MultiLineString, GeometryCollection)):
        lines = []
        for g in geom.geoms:
            lines.extend(flatten_lines(g))
        return lines
    return []   # 其它类型忽略

def png_to_svg(png_path: Path):
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(png_path)

    # 1. 二值化 & 闭运算
    _, bw = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. Hough 线段
    raw = cv2.HoughLinesP(bw, 1, np.pi/180, threshold=120,
                          minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)
    if raw is None:
        raise RuntimeError("未检测到线段，请调阈值或预处理图像")

    # 3. 合并重叠线
    line_objs = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in raw[:, 0]]
    merged = linemerge(unary_union(line_objs))
    merged_lines = flatten_lines(merged)       # 保证全是 LineString

    # 4. 输出 SVG
    svg_path = png_path.with_suffix('.svg')
    dwg = svgwrite.Drawing(str(svg_path),
                           size=(f'{img.shape[1]}px', f'{img.shape[0]}px'),
                           profile='tiny')
    for ls in merged_lines:
        x1, y1, x2, y2 = map(int, (*ls.coords[0], *ls.coords[-1]))
        dwg.add(dwg.line((x1, y1), (x2, y2), stroke='black', stroke_width=1))
    dwg.save()
    print(f'✔ 已生成 {svg_path}')

if __name__ == '__main__':
    png_file = Path(r'E:\code\CubiCasa5k\DataPreparation\literature_png\heatmap_fig4.png')
    png_to_svg(png_file)

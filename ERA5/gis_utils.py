"""
gis_utils.py
-------------
Mô-đun tiện ích GIS cho đồ án Rainfall Nowcasting (ERA5).
Cung cấp ranh giới hành chính Việt Nam (34 đơn vị hành chính tỉnh/thành phố),
đường biên giới quốc gia, các nước láng giềng khu vực Đông Nam Á, và
đường bao (bounding boxes) + nhãn cho các đảo và quần đảo chủ quyền của
Việt Nam (Hoàng Sa, Trường Sa, Phú Quốc, Côn Đảo, Bạch Long Vĩ...).
"""

import os
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Thư mục chứa dữ liệu GIS
GIS_DIR = Path(__file__).parent / "gis_data"
VN_PROVINCES_FILE = GIS_DIR / "vietnam_provinces.geojson"
NE_COUNTRIES_FILE = GIS_DIR / "ne_countries.geojson"

# Toạ độ bounding box và thông tin các quần đảo / đảo trọng yếu của Việt Nam
VIETNAM_ISLANDS_BOUNDS = {
    "hoang_sa": {
        "name": "Quần đảo Hoàng Sa (Đà Nẵng)",
        "short_name": "QĐ. Hoàng Sa",
        "bbox": [111.0, 15.5, 113.2, 17.5],  # [lon_min, lat_min, lon_max, lat_max]
        "center": [112.1, 16.5],
        "label_pos": [113.5, 16.8],
        "color": "#D90429",
    },
    "truong_sa": {
        "name": "Quần đảo Trường Sa (Khánh Hòa)",
        "short_name": "QĐ. Trường Sa",
        "bbox": [111.5, 6.5, 117.5, 12.5],  # [lon_min, lat_min, lon_max, lat_max]
        "center": [114.5, 9.5],
        "label_pos": [117.8, 10.0],
        "color": "#D90429",
    },
    "phu_quoc": {
        "name": "Đảo Phú Quốc (Kiên Giang)",
        "short_name": "Đ. Phú Quốc",
        "bbox": [103.7, 9.8, 104.2, 10.5],
        "center": [103.95, 10.15],
        "label_pos": [102.5, 10.2],
        "color": "#FF8800",
    },
    "con_dao": {
        "name": "Côn Đảo (Bà Rịa - Vũng Tàu)",
        "short_name": "Côn Đảo",
        "bbox": [106.5, 8.55, 106.75, 8.85],
        "center": [106.6, 8.7],
        "label_pos": [107.0, 8.4],
        "color": "#FF8800",
    },
    "bach_long_vi": {
        "name": "Đảo Bạch Long Vĩ (Hải Phòng)",
        "short_name": "Đ. Bạch Long Vĩ",
        "bbox": [107.65, 20.1, 107.85, 20.25],
        "center": [107.75, 20.17],
        "label_pos": [108.2, 20.3],
        "color": "#FF8800",
    }
}

# Cache GeoDataFrames để tránh load lại nhiều lần
_GDF_VN = None
_GDF_WORLD = None


def load_gis_layers():
    """Tải các lớp GeoPandas cho Việt Nam và các nước lân cận."""
    global _GDF_VN, _GDF_WORLD
    
    if _GDF_VN is None and VN_PROVINCES_FILE.exists():
        try:
            _GDF_VN = gpd.read_file(VN_PROVINCES_FILE)
        except Exception as e:
            print(f"Cảnh báo: Không load được file {VN_PROVINCES_FILE}: {e}")
            _GDF_VN = None

    if _GDF_WORLD is None and NE_COUNTRIES_FILE.exists():
        try:
            _GDF_WORLD = gpd.read_file(NE_COUNTRIES_FILE)
        except Exception as e:
            print(f"Cảnh báo: Không load được file {NE_COUNTRIES_FILE}: {e}")
            _GDF_WORLD = None

    return _GDF_VN, _GDF_WORLD


def draw_vietnam_map(
    ax,
    extent=[100.0, 120.0, 5.0, 25.0],
    show_provinces=True,
    show_islands=True,
    show_islands_box=True,
    show_sea_bounds=True,
    show_country_labels=True,
    province_edgecolor="#4A4E69",
    province_linewidth=0.6,
    province_alpha=0.85,
    country_edgecolor="#22223B",
    country_linewidth=1.2,
    island_box_style="dashed",
):
    """
    Vẽ lớp bản đồ Việt Nam (tỉnh thành, biên giới) và bao bọc các quần đảo Hoàng Sa, Trường Sa.
    
    Tham số:
    ---------
    ax : matplotlib.axes.Axes
        Trục vẽ của biểu đồ.
    extent : list [lon_min, lon_max, lat_min, lat_max]
        Phạm vi kinh độ / vĩ độ của bản đồ.
    show_provinces : bool
        Có hiển thị ranh giới 34 đơn vị hành chính Việt Nam hay không.
    show_islands : bool
        Có đánh dấu và hiển thị nhãn các đảo/quần đảo hay không.
    show_islands_box : bool
        Có vẽ khung bao bọc (bounding box) cho Hoàng Sa và Trường Sa hay không.
    show_sea_bounds : bool
        Có hiển thị vùng phân định lãnh hải đại diện hay không.
    show_country_labels : bool
        Có hiển thị tên các nước láng giềng hay không.
    """
    gdf_vn, gdf_world = load_gis_layers()
    
    lon_min, lon_max, lat_min, lat_max = extent

    # 1. Vẽ ranh giới các nước khu vực Đông Nam Á
    if gdf_world is not None:
        gdf_world.boundary.plot(
            ax=ax,
            edgecolor=country_edgecolor,
            linewidth=country_linewidth * 0.7,
            linestyle="-",
            alpha=0.6,
            zorder=3
        )
    
    # 2. Vẽ ranh giới 34 đơn vị hành chính Việt Nam
    if gdf_vn is not None and show_provinces:
        gdf_vn.boundary.plot(
            ax=ax,
            edgecolor=province_edgecolor,
            linewidth=province_linewidth,
            linestyle="-",
            alpha=province_alpha,
            zorder=4
        )
        
        # Vẽ đường bao quốc gia Việt Nam đậm hơn
        try:
            vn_union = gdf_vn.unary_union
            gpd.GeoSeries([vn_union]).boundary.plot(
                ax=ax,
                edgecolor="#B7094C",
                linewidth=country_linewidth,
                alpha=0.9,
                zorder=5
            )
        except Exception:
            pass

    # 3. Vẽ đường bao (Bounding Box) và Nhãn chủ quyền cho Quần đảo Hoàng Sa & Trường Sa
    if show_islands:
        for key, island in VIETNAM_ISLANDS_BOUNDS.items():
            bbox = island["bbox"]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            center = island["center"]
            
            # Kiểm tra xem đảo có nằm trong khung nhìn extent không
            if not (bbox[2] < lon_min or bbox[0] > lon_max or bbox[3] < lat_min or bbox[1] > lat_max):
                if show_islands_box and key in ["hoang_sa", "truong_sa"]:
                    # Vẽ khung chữ nhật nổi bật bao bọc toàn bộ quần đảo
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]),
                        w,
                        h,
                        linewidth=1.3,
                        edgecolor=island["color"],
                        facecolor="none",
                        linestyle="--" if island_box_style == "dashed" else "-",
                        alpha=0.85,
                        zorder=6
                    )
                    ax.add_patch(rect)
                    
                    # Thêm nền nhẹ cho vùng quần đảo để dễ quan sát
                    rect_fill = patches.Rectangle(
                        (bbox[0], bbox[1]),
                        w,
                        h,
                        facecolor=island["color"],
                        alpha=0.03,
                        zorder=2
                    )
                    ax.add_patch(rect_fill)

                # Vẽ điểm tâm đảo / biểu tượng
                ax.plot(
                    center[0],
                    center[1],
                    marker="*",
                    markersize=9 if key in ["hoang_sa", "truong_sa"] else 6,
                    color=island["color"],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=7
                )
                
                # Vẽ text nhãn
                label_text = island["name"] if key in ["hoang_sa", "truong_sa"] else island["short_name"]
                pos = island["label_pos"]
                
                # Căn chỉnh vị trí nhãn nếu nằm ngoài màn hình
                pos_x = min(max(pos[0], lon_min + 0.5), lon_max - 3.5)
                pos_y = min(max(pos[1], lat_min + 0.5), lat_max - 0.5)
                
                ax.annotate(
                    label_text,
                    xy=(center[0], center[1]),
                    xytext=(pos_x, pos_y),
                    fontsize=8.5 if key in ["hoang_sa", "truong_sa"] else 7.5,
                    fontweight="bold" if key in ["hoang_sa", "truong_sa"] else "normal",
                    color=island["color"],
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=island["color"], alpha=0.85, lw=0.8),
                    arrowprops=dict(arrowstyle="->", color=island["color"], lw=0.8, alpha=0.8) if key not in ["hoang_sa", "truong_sa"] else None,
                    zorder=8
                )

    # 4. Hiển thị nhãn tên các nước xung quanh
    if show_country_labels:
        neighbors = [
            ("LÀO (LAOS)", [103.5, 18.5]),
            ("CAMPUCHIA", [104.8, 12.5]),
            ("THÁI LAN", [101.0, 15.0]),
            ("TRUNG QUỐC", [106.0, 23.5]),
            ("PHILIPPINES", [119.0, 14.5]),
            ("BIỂN ĐÔNG", [113.0, 14.0]),
            ("VỊNH BẮC BỘ", [107.5, 19.5]),
            ("VỊNH THÁI LAN", [101.5, 9.5]),
        ]
        for name, (c_lon, c_lat) in neighbors:
            if lon_min <= c_lon <= lon_max and lat_min <= c_lat <= lat_max:
                is_sea = "BIỂN" in name or "VỊNH" in name
                ax.text(
                    c_lon,
                    c_lat,
                    name,
                    fontsize=8.5 if is_sea else 8.0,
                    fontweight="bold" if is_sea else "normal",
                    fontstyle="italic" if is_sea else "normal",
                    color="#0077B6" if is_sea else "#6C757D",
                    alpha=0.75,
                    ha="center",
                    va="center",
                    zorder=5
                )

    # Thiết lập giới hạn trục và định dạng toạ độ
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Kinh độ (°E)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Vĩ độ (°N)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, color="#8D99AE")
    
    # Định dạng ticks
    ax.set_xticks(np.arange(np.ceil(lon_min), np.floor(lon_max) + 1, 2.5))
    ax.set_yticks(np.arange(np.ceil(lat_min), np.floor(lat_max) + 1, 2.5))

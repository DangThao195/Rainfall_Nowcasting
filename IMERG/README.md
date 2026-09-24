# IMERG Rainfall Data Pipeline

Dữ liệu IMERG cho bài toán dự báo mưa định lượng (rainfall nowcasting) tại Việt Nam.

## 1. Phạm vi dữ liệu

Nguồn dữ liệu là NASA GPM IMERG Final Run:

- Collection: `GPM_3IMERGHH`
- Version: `07`
- Tần suất: 30 phút/frame
- Biến chính: `rainfall(time, lat, lon)`
- Đơn vị: `mm/30 min`
- Bounding box: `102-110°E`, `8-24.5°N`
- Grid: `165 x 80`

Bounding box bao quanh Việt Nam nên vẫn chứa biển và vùng lân cận. Land mask polygon Việt Nam được tạo riêng trong pipeline data.

## 2. Dữ liệu hiện có

| File | Khoảng thời gian | Số frame | Trạng thái |
|---|---|---:|---|
| `imerg_vietnam_2023_2024.nc` | `2023-01-01 00:00` đến `2025-01-01 00:00` | 35,089 | Có frame biên `2025-01-01`; pipeline loại trùng khi nối file |
| `imerg_vietnam_2025_2026.nc` | `2025-01-01 00:00` đến `2025-09-30 23:30` | 13,104 | Chưa đủ toàn bộ 2025-2026 |

Phần sau `2025-09-30 23:30` hiện chưa có trong file 2025-2026. Không được coi phần thiếu là mưa bằng `0`; pipeline chỉ dùng timestamp thực sự tồn tại.

Mỗi frame có `165 x 80 = 13,200` pixel. Interface nowcasting:

- Input: `(6, 165, 80, 1)` tương đương 3 giờ lịch sử.
- Target: `(4, 165, 80, 1)` tương đương 2 giờ dự báo.

## 3. Notebook

### Download và crop 2023-2024

[dowload_imerg_2023_2024.ipynb](dowload_imerg_2023_2024.ipynb)

Tìm granule bằng `earthaccess`, tải theo batch, đọc HDF5, crop bounding box, chuyển rainfall sang `mm/30 min`, ghi NetCDF và resume từ các frame đã có.

### Download và crop 2025-2026

[dowload_imerg_2025_2026.ipynb](dowload_imerg_2025_2026.ipynb)

Notebook này dùng output riêng `imerg_vietnam_2025_2026.nc`. Khi resume, nó đọc timestamp hiện có và tìm phần còn thiếu. Dữ liệu chưa được NASA cung cấp không được tự tạo.

### EDA

- [eda_imerg_2023_2024.ipynb](eda_imerg_2023_2024.ipynb)
- [eda_imerg_2025_2026.ipynb](eda_imerg_2025_2026.ipynb)

EDA kiểm tra integrity, timestamp thiếu/trùng, NaN, phân bố rainfall, bản đồ không gian, seasonality và continuity của cửa sổ 6+4.

### Chuẩn bị dữ liệu cho model

[run.ipynb](run.ipynb)

Notebook này:

1. Nối hai NetCDF và loại timestamp trùng.
2. Chia dữ liệu theo thời gian.
3. Tạo Vietnam land mask và trọng số đất liền.
4. Kiểm tra sliding window 6 frame input + 4 frame target.
5. Chuẩn hóa bằng `log1p` và z-score theo thống kê của train.
6. Lưu split, window index và metadata.

## 4. Chiến lược train/validation/test

Pipeline dùng split theo thời gian, không xáo trộn ngẫu nhiên:

| Split | Khoảng thời gian | Vai trò |
|---|---|---|
| Train | `2023-01-01` đến trước `2025-01-01` | Huấn luyện model |
| Validation | `2025-01-01` đến trước `2025-07-01` | Chọn hyperparameter và theo dõi overfitting |
| Test | Từ `2025-07-01` đến frame cuối thực tế đã tải | Đánh giá sau cùng |

Test hiện chỉ có phần `2025-07-01` đến `2025-09-30 23:30`. Khi tải thêm dữ liệu, chạy lại `run.ipynb` để cập nhật test và metadata.

Một sample chỉ hợp lệ khi toàn bộ 10 frame liên tục, cách nhau 30 phút, không có NaN và không vượt ranh giới split.

## 5. Chuẩn hóa

Với rainfall thô `x` theo `mm/30 min`:

1. Log transform: `x_log = log1p(x)`.
2. Z-score: `z = (x_log - train_log_mean) / train_log_std`.

`train_log_mean` và `train_log_std` chỉ được tính từ train. Validation và test dùng lại hai giá trị này để tránh data leakage.

Khôi phục về rainfall gốc:

```python
rainfall = np.expm1(z * train_log_std + train_log_mean)
```

## 6. Vietnam land mask

Land mask dùng polygon Việt Nam từ Natural Earth và tâm pixel của grid IMERG:

- `land_mask = 1` trong polygon Việt Nam, `0` ngoài polygon.
- `W_land = 2.5` trên đất liền Việt Nam, `1.0` ngoài đất liền.

Các file được tạo trong `data/prepared/`:

- `vietnam_land_mask.pt`
- `vietnam_land_weight.pt`
- `vietnam_land_mask.nc`
- `vietnam_land_weight.nc`

Kích thước mask là `(165, 80)`, dùng cho Vietnam Land-Weighted L1 loss.

## 7. Output của run.ipynb

```text
IMERG/data/prepared/
├── imerg_train_raw.nc
├── imerg_validation_raw.nc
├── imerg_test_raw.nc
├── imerg_train_normalized.nc
├── imerg_validation_normalized.nc
├── imerg_test_normalized.nc
├── train_window_starts.npy
├── validation_window_starts.npy
├── test_window_starts.npy
├── vietnam_land_mask.pt
├── vietnam_land_weight.pt
├── vietnam_land_mask.nc
├── vietnam_land_weight.nc
└── dataset_metadata.json
```

## 8. Cách chạy

### Google Colab

1. Mở notebook download tương ứng và mount Google Drive.
2. Chạy download/crop nếu cần cập nhật dữ liệu.
3. Chạy notebook EDA để kiểm tra integrity.
4. Chạy `run.ipynb` từ đầu đến cuối để tạo split, land mask và dữ liệu normalized.

### Local

Đặt hai file NetCDF ở thư mục `IMERG/` hoặc cấu hình lại đường dẫn trong cell đầu của `run.ipynb`.

## 9. Kiểm tra trước khi train

- Kiểm tra timestamp thiếu/trùng và NaN.
- Xác nhận `dataset_metadata.json` ghi đúng coverage thực tế.
- Không điền `0` cho giai đoạn chưa tải.
- Không tính normalization statistics từ validation/test.
- Xác nhận số valid window trước khi khởi tạo DataLoader.
- Xác nhận model nhận input `(B, 6, 165, 80, 1)` và target `(B, 4, 165, 80, 1)`.

## 10. Git và dữ liệu

Các file NetCDF đã crop và artifact trong `data/prepared/` được phép commit nếu kích thước repository phù hợp. Raw HDF5, file batch tạm, figures, cache và failure logs vẫn bị bỏ qua trong `.gitignore`.

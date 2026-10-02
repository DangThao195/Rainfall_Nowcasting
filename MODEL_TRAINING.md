# Huấn luyện IMERG ConvLSTM

Pipeline dùng 6 frame IMERG gần nhất (3 giờ) để dự báo 4 frame tiếp theo (2 giờ). Mỗi frame có kích thước `165 x 80`, đơn vị kết quả là `mm/30 min`.

## Thành phần

- Dataset NetCDF được đọc theo từng window, không nạp cả split vào RAM.
- ConvLSTM encoder-decoder dự báo tự hồi quy bốn bước.
- Weighted Huber loss dùng `W_land=2.5` trong lãnh thổ Việt Nam và `1.0` ngoài lãnh thổ; pixel có mưa từ `0.1 mm/30 min` được tăng trọng số.
- Persistence baseline lặp lại frame quan sát cuối cùng.
- Báo cáo MAE, RMSE, CSI tại `0.1/2.5/10 mm` và FSS cửa sổ `3x3/5x5`, gồm tổng hợp và từng lead time.

## Chạy kiểm tra nhanh tại máy local

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m rainfall_nowcasting.train `
  --epochs 1 `
  --batch-size 1 `
  --hidden-channels 2 4 `
  --max-train-batches 1 `
  --max-val-batches 1 `
  --max-test-batches 1 `
  --output-dir IMERG_data/outputs/smoke
```

## Huấn luyện đầy đủ trên Google Colab

1. Bật GPU trong `Runtime > Change runtime type`.
2. Mount Google Drive và đặt repo ở `/content/Rainfall_Nowcasting` hoặc clone repo vào đó.
3. Dữ liệu prepared có thể nằm trên Drive; truyền đúng đường dẫn qua `--data-dir`.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content/Rainfall_Nowcasting
!pip install -q netCDF4
!python -m rainfall_nowcasting.train \
  --data-dir "/content/drive/MyDrive/Rainfall_Nowcasting/IMERG_data/prepared" \
  --output-dir "/content/drive/MyDrive/Rainfall_Nowcasting/outputs/convlstm" \
  --epochs 30 \
  --batch-size 4 \
  --num-workers 2 \
  --hidden-channels 16 32
```

Nếu hết VRAM, giảm `--batch-size` xuống `2` hoặc `1`. Nếu còn nhiều VRAM, thử `--hidden-channels 32 64`.

## Đánh giá lại checkpoint

```bash
python -m rainfall_nowcasting.evaluate \
  --data-dir IMERG_data/prepared \
  --checkpoint IMERG_data/outputs/convlstm/best.pt \
  --batch-size 4
```

Kết quả được lưu tại `test_metrics.json`. `convlstm` và `persistence` dùng chính xác cùng test windows và cùng land mask.

## Test và visualize trên máy local

Tải `best.pt` từ Google Drive về:

```text
IMERG_data/outputs/convlstm/best.pt
```

Đánh giá toàn bộ test split bằng CPU:

```powershell
.\.venv\Scripts\python.exe -m rainfall_nowcasting.evaluate `
  --data-dir IMERG_data\prepared `
  --checkpoint IMERG_data\outputs\convlstm\best.pt `
  --output IMERG_data\outputs\convlstm\test_metrics.json `
  --batch-size 4 `
  --device cpu
```

Tự chọn window test có lượng mưa trung bình trên Việt Nam lớn nhất và vẽ ba
cột: ground truth, dự báo và chênh lệch `predicted - truth`:

```powershell
.\.venv\Scripts\python.exe -m rainfall_nowcasting.visualize `
  --data-dir IMERG_data\prepared `
  --checkpoint IMERG_data\outputs\convlstm\best.pt `
  --device cpu
```

Muốn xem một window cụ thể, thêm `--sample-index 1000`. Thêm `--show-full-grid` nếu muốn xem toàn bộ bounding box thay vì chỉ mask Việt Nam.

## Dự báo cuốn chiếu cho cả ngày tiếp theo

File `visualize_day_ahead.py` là công cụ riêng, không thay thế hoặc ghi đè
`visualize.py`. Ví dụ: đọc đủ ngày 2025-09-28 trong test split và dự báo 48 mốc
nửa giờ của ngày 2025-09-29:

```powershell
.\.venv\Scripts\python.exe -m rainfall_nowcasting.visualize_day_ahead `
  --data-dir IMERG_data\prepared `
  --checkpoint IMERG_data\outputs\convlstm\best.pt `
  --split test `
  --date 2025-09-28 `
  --device cpu
```

Mỗi lần chạy tạo một thư mục `day_ahead_visualizations` chứa ảnh so sánh các
mốc trong ngày, tổng lượng mưa ngày, chuỗi thời gian trung bình trên Việt Nam và
file JSON metric. Hai ngày liên tiếp phải nằm trọn trong split đã chọn.

Lưu ý: checkpoint hiện tại được train theo bài toán 6 frame đầu vào → 4 frame
đầu ra (3 giờ → 2 giờ). Dự báo 24 giờ phải lặp mô hình 12 lần và dùng dự báo cũ
làm đầu vào mới, nên sai số có thể tích lũy mạnh. Kết quả này phù hợp để khảo sát,
không nên xem là dự báo ngày nghiệp vụ nếu chưa train lại mô hình cho horizon dài.

## File output

- `best.pt`: checkpoint có validation loss tốt nhất.
- `last.pt`: trạng thái epoch cuối.
- `history.json`: lịch sử loss.
- `loss_curve.png`: đồ thị train/validation loss.
- `test_metrics.json`: kết quả ConvLSTM và persistence trên test split.

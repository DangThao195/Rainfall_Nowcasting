# Khảo sát dữ liệu Himawari cho rainfall nowcasting tại Việt Nam

Ngày khảo sát: 2026-09-11  
Phạm vi: Việt Nam và Đà Nẵng  
Tần suất hệ thống cần: 30 phút hoặc 1 giờ

## 1. Kết luận đề xuất

Himawari đáp ứng tốt vai trò nguồn quan sát mây/khí quyển gần thời gian thực cho đề tài. Vệ tinh đang vận hành là **Himawari-9**; Himawari-8 là vệ tinh dự phòng. AHI quan sát Full Disk mỗi 10 phút với 16 băng phổ, nên hệ thống có thể lấy mẫu xuống 30 phút hoặc 1 giờ mà không làm thay đổi nguồn gốc dữ liệu.

Phương án nên dùng cho project:

- Nguồn chính: bucket công khai `noaa-himawari9` thuộc NOAA Open Data Dissemination trên AWS.
- Nhịp lấy mẫu mặc định: **30 phút**, tại phút `00` và `30` mỗi giờ UTC, để khớp dữ liệu IMERG `GPM_3IMERGHH` đang có trong project.
- Tập băng khởi đầu: `B08`, `B09`, `B10`, `B13`, `B14`, `B15`; đây là các băng hồng ngoại/hơi nước dùng được cả ngày lẫn đêm.
- Phạm vi tải cho Việt Nam: chỉ lấy các segment `S03`, `S04`, `S05`, rồi giải mã và crop về bbox hiện tại của project: `102–110°E, 8–24.5°N`.
- Phạm vi Đà Nẵng: tạo ROI từ cùng frame Việt Nam; không chạy một crawler riêng. Đà Nẵng nằm trong segment `S04`.
- Sau xử lý: resample trực tiếp về đúng lưới `lat/lon` của NetCDF IMERG, lưu NetCDF hoặc Zarr; chỉ giữ file `.DAT.bz2` thô trong một cửa sổ ngắn để retry.

Ở cấu hình sáu băng và ba segment trên, một frame thực tế được kiểm tra ngày 2026-09-11 có dung lượng tải khoảng **38.09 MiB**. Nhịp 30 phút tương đương khoảng **1.79 GiB/ngày** trước khi crop; nhịp 1 giờ khoảng **0.89 GiB/ngày**. Dung lượng file sau crop sẽ nhỏ hơn đáng kể.

## 2. Đặc tính dữ liệu

Theo JMA, AHI trên Himawari-8/9 có 16 băng, quan sát Full Disk mỗi 10 phút. Độ phân giải danh định tại điểm dưới vệ tinh là:

| Nhóm băng | Độ phân giải danh định | Tần suất Full Disk |
|---|---:|---:|
| B03 | 0.5 km | 10 phút |
| B01, B02, B04 | 1 km | 10 phút |
| B05–B16 | 2 km | 10 phút |

Nguồn: [JMA – AHI observation area and periodicity](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html) và [Himawari Standard Data User's Guide](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v13.pdf).

### Tập băng đề xuất ban đầu

| Băng | Miền phổ gần đúng | Thông tin hữu ích |
|---|---:|---|
| B08 | 6.2 µm | Hơi nước tầng cao, cấu trúc đối lưu |
| B09 | 6.9 µm | Hơi nước tầng giữa |
| B10 | 7.3 µm | Hơi nước tầng thấp/trung |
| B13 | 10.4 µm | Nhiệt độ sáng cửa sổ IR, đỉnh mây |
| B14 | 11.2 µm | Cửa sổ IR, theo dõi mây |
| B15 | 12.4 µm | Split-window, hỗ trợ phân biệt mây/bề mặt |

Các giá trị đầu vào sau hiệu chỉnh nên được lưu dưới dạng brightness temperature theo Kelvin. Có thể bổ sung `B07`, `B11`, `B12`, `B16` trong thí nghiệm ablation. `B03` có độ phân giải cao nhưng chỉ hữu ích vào ban ngày và làm tăng mạnh băng thông, vì vậy không nên đưa vào baseline 24/7 ngay từ đầu.

Lưu ý quan trọng: Himawari không đo trực tiếp lượng mưa tại mặt đất. Các băng AHI và sản phẩm mây là **predictor**; IMERG/radar/trạm mưa vẫn là nhãn hoặc nguồn quan sát mưa.

## 3. So sánh các nguồn truy cập

| Nguồn | Truy cập | Độ trễ/tần suất | Điểm mạnh | Hạn chế | Kết luận |
|---|---|---|---|---|---|
| NOAA/AWS `noaa-himawari9` | Công khai, không cần AWS account; hỗ trợ `--no-sign-request` | Full Disk 10 phút; có archive từ 2015 | Raw HSD đầy đủ, archive và near-real-time cùng một cấu trúc; có SNS notification | File HSD chia theo lát Bắc–Nam, không thể tải trực tiếp theo bbox Việt Nam bên trong file nén | **Nguồn chính nên dùng** |
| JAXA P-Tree | Phải đăng ký tài khoản; xét duyệt có thể mất vài ngày | HSD near-real-time khoảng 5–20 phút; Full Disk 10 phút | Có HSD, L1 gridded NetCDF và sản phẩm vật lý mây | Điều khoản hạn chế tái phân phối; cloud property chỉ ban ngày; file Full Disk lớn | Nguồn đối chứng/nghiên cứu bổ sung |
| JMA HimawariCloud | Chỉ dành cho cơ quan khí tượng thủy văn quốc gia trong vùng phủ | Full Disk 10 phút; giữ 72 giờ | 16 băng độ phân giải đầy đủ, dịch vụ vận hành chính thức | Không phù hợp tài khoản project thông thường; toàn bộ HSD khoảng 103 GB/ngày | Không chọn cho PoC |
| JMA Real-Time Image | HTTP/JPEG công khai | Ảnh khu vực mỗi 10 phút | Rất nhẹ; khu vực Southeast Asia 1 phủ `80–115°E, 0–30°N` nên chứa toàn bộ Việt Nam | Là ảnh đã render, không phải số liệu khoa học gốc; không phù hợp huấn luyện | Dùng cho màn hình quick-look |
| NICT Himawari Real-time Web | Web/ảnh; tải dữ liệu chính thức cần tài khoản | Full Disk khoảng 20 phút trễ | Zoom và animation có sẵn | Giới hạn mục đích sử dụng/ghi nguồn; giao diện có thể thay đổi | Chỉ dùng tham khảo/đối chiếu |

Tài liệu nguồn:

- [NOAA/AWS Registry – JMA Himawari-8/9](https://registry.opendata.aws/noaa-himawari/)
- [JAXA P-Tree registration and latency](https://www.eorc.jaxa.jp/ptree/registration_top.html)
- [JAXA P-Tree user's guide](https://www.eorc.jaxa.jp/ptree/userguide.html)
- [JMA HimawariCloud](https://www.data.jma.go.jp/mscweb/en/himawari89/cloud_service/cloud_service.html)
- [JMA Himawari Real-Time Image](https://www.data.jma.go.jp/mscweb/data/himawari/index.html)
- [NICT Himawari Real-time Web help](https://himawari8.nict.go.jp/himawari8-help.htm)

## 4. Khả năng crawl gần thời gian thực

### 4.1 Cấu trúc bucket đề xuất

Bucket:

```text
s3://noaa-himawari9/
```

Prefix Full Disk Level-1b:

```text
AHI-L1b-FLDK/YYYY/MM/DD/HHMM/
```

Ví dụ object:

```text
AHI-L1b-FLDK/2026/09/11/0510/
HS_H09_20260911_0510_B13_FLDK_R20_S0410.DAT.bz2
```

Mỗi thời điểm Full Disk có 16 băng × 10 segment = 160 object. Với baseline đề xuất, crawler chỉ cần 6 băng × 3 segment = 18 object cho mỗi frame Việt Nam.

### 4.2 Lịch crawl 30 phút

Crawler nên chạy polling mỗi 5 phút nhưng chỉ nhận các observation slot `HH:00` và `HH:30`:

1. Tính các slot đã qua ít nhất 20 phút trong cửa sổ look-back 90 phút.
2. Liệt kê prefix tương ứng trong S3.
3. Chỉ đánh dấu frame hoàn chỉnh khi đủ 18 object yêu cầu.
4. Tải các object chưa có, kiểm tra kích thước/ETag và giải nén.
5. Đọc HSD, hiệu chỉnh, ghép segment, crop và resample.
6. Ghi output theo timestamp UTC bằng thao tác atomic.
7. Cập nhật manifest; xóa raw sau khi output đã kiểm tra thành công.

Nếu chọn nhịp 1 giờ, chỉ đổi slot thành `HH:00`; toàn bộ pipeline còn lại giữ nguyên.

Không nên chỉ hỏi “file mới nhất” vì dễ bỏ frame khi mạng lỗi. Cơ chế look-back + manifest giúp crawler idempotent, tự bù dữ liệu thiếu và không tạo bản ghi trùng.

### 4.3 Độ trễ quan sát được

Trong kiểm tra trực tiếp lúc **05:28 UTC ngày 2026-09-11**:

- Prefix Full Disk mới nhất là `05:10`.
- Các object của scan `05:10` được ghi vào bucket trong khoảng khoảng `05:20–05:24 UTC`.
- Độ trễ thực tế quan sát được khoảng 10–14 phút tính từ mốc nominal, phù hợp cho lịch xử lý sau 20 phút.

Đây là một phép đo tại một thời điểm, không phải SLA. Pipeline nên đặt retry và theo dõi latency p50/p95.

## 5. Phạm vi không gian

### 5.1 Việt Nam

Giữ bbox đang dùng trong project IMERG:

```text
longitude: 102–110°E
latitude:    8–24.5°N
```

Theo phép chiếu geostationary của HSD, bbox này giao với ba segment Full Disk:

```text
S03, S04, S05
```

Vì mỗi segment vẫn kéo dài toàn chiều ngang của Full Disk, hệ thống phải tải đủ ba lát rồi mới crop theo kinh độ. Sau khi reproject, nên lưu đúng grid của IMERG thay vì tạo một grid Himawari riêng; điều này giảm sai lệch khi ghép tensor theo thời gian.

### 5.2 Đà Nẵng

Đà Nẵng nằm trong `S04`. ROI Đà Nẵng nên được cắt từ sản phẩm Việt Nam đã xử lý, nên không phát sinh thêm băng thông tải nguồn.

Cần phiên bản hóa polygon hành chính. Từ năm 2025, thành phố Đà Nẵng mới bao gồm toàn bộ Đà Nẵng cũ và tỉnh Quảng Nam cũ; vì vậy polygon “Đà Nẵng” cũ và mới khác nhau rất lớn. Hệ thống nên lưu tối thiểu:

```text
roi_id: danang_admin_2025
boundary_version: 2025-07-01
```

Nếu mục tiêu thực tế chỉ là vùng đô thị ven biển Đà Nẵng cũ, nên tạo thêm ROI riêng như `danang_urban_legacy` thay vì gọi chung là Đà Nẵng.

Việt Nam hiện có 34 đơn vị hành chính cấp tỉnh sau Nghị quyết 202/2025/QH15. Bản đồ hiển thị tỉnh/thành cần dùng lớp ranh giới sau sáp nhập, không dùng bộ 63 tỉnh cũ. Nguồn xác nhận: [Cổng Thông tin điện tử Chính phủ – bản đồ hành chính 34 tỉnh/thành](https://baochinhphu.vn/tra-cuu-dia-gioi-sau-sap-nhap-qua-ban-do-dien-tu-102250708175104468.htm) và [Nghị quyết 202/2025/QH15](https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-quyet-so-202-2025-qh15-ve-sap-xep-don-vi-hanh-chinh-cap-tinh-119250612174148722.htm).

## 6. Khả năng visualize

### 6.1 Notebook/EDA

Luồng phù hợp với notebook IMERG hiện có:

```text
HSD -> Satpy -> resample sang lat/lon -> xarray -> Cartopy/Matplotlib
                                      -> overlay polygon Việt Nam/tỉnh/Đà Nẵng
```

[Satpy có reader `ahi_hsd`](https://satpy.readthedocs.io/en/stable/api/satpy.readers.ahi_hsd.html) để đọc HSD, hiệu chỉnh radiance/brightness temperature và giữ metadata phép chiếu. Không nên vẽ trực tiếp ma trận pixel HSD bằng `imshow` rồi gán bbox vì ảnh gốc dùng phép chiếu địa tĩnh, không phải lưới kinh-vĩ tuyến đều.

Các lớp nên hiển thị:

- B13 brightness temperature dạng thang xám hoặc màu nhiệt.
- Chênh lệch `B13-B15` hoặc tổ hợp RGB mây nếu cần.
- Biên giới Việt Nam, ranh giới 34 tỉnh/thành và polygon Đà Nẵng.
- Nhãn timestamp UTC và giờ Việt Nam.
- Animation 6–12 frame để quan sát chuyển động/phát triển mây.

### 6.2 Web map

Sau khi xử lý, có hai lựa chọn:

- Xuất PNG đã tô màu kèm geographic bounds, hiển thị bằng Leaflet/MapLibre và overlay GeoJSON ranh giới.
- Xuất GeoTIFF/Cloud Optimized GeoTIFF hoặc tile XYZ nếu cần zoom nhiều mức và truy vấn giá trị pixel.

Để làm quick-look trước khi pipeline raw hoàn thiện, có thể dùng JPEG `Southeast Asia 1` của JMA. Một ảnh B13 được kiểm tra có kích thước khoảng **169 KiB**, rất nhẹ để refresh 30 phút/lần. Tuy nhiên ảnh này không thay thế dữ liệu HSD/NetCDF dùng cho mô hình.

## 7. Ước lượng băng thông và lưu trữ

Số liệu dưới đây lấy từ một frame Himawari-9 thực tế ngày 2026-09-11, gồm 6 băng `B08/B09/B10/B13/B14/B15` và segment `S03/S04/S05`. Dung lượng thay đổi theo nội dung ảnh và tỷ lệ nén.

| Nhịp lấy mẫu | Frame/ngày | Tải raw/ngày | Tải raw/30 ngày | Tải raw/năm |
|---|---:|---:|---:|---:|
| 30 phút | 48 | ~1.79 GiB | ~54 GiB | ~653 GiB |
| 1 giờ | 24 | ~0.89 GiB | ~27 GiB | ~326 GiB |

Khuyến nghị lưu trữ:

- Raw `.DAT.bz2`: rolling cache 1–3 ngày.
- Dữ liệu đã crop/resample: NetCDF hoặc Zarr, chunk theo `time` và `channel`.
- Quick-look: PNG/JPEG, chỉ giữ theo chính sách hiển thị.
- Manifest: timestamp, object key, ETag, size, trạng thái xử lý, latency và lỗi.

## 8. Đồng bộ với IMERG

IMERG trong project đang có timestep 30 phút và cửa sổ nowcasting theo các mốc `t-150 ... t` rồi `t+30 ... t+120`. Vì vậy 30 phút là lựa chọn tốt hơn 1 giờ cho baseline.

Các quy tắc đồng bộ:

- Chuẩn hóa toàn bộ timestamp về UTC.
- Lấy Himawari đúng các slot `:00` và `:30`, không nội suy nếu chưa cần thiết.
- Ghi rõ dùng nominal scan time hay scan midpoint; Full Disk được quét trong một khoảng thời gian chứ không chụp tức thời.
- Reindex Himawari theo trục thời gian IMERG và lưu cờ missing cho từng channel/frame.
- Không biến missing thành 0; 0 có ý nghĩa vật lý khác với thiếu dữ liệu.
- Split train/validation/test theo thời gian trước khi chuẩn hóa để tránh leakage.

## 9. Rủi ro và lưu ý sử dụng

- **Đây là near-real-time, không phải hard real-time:** không có SLA cho bucket công khai; cần retry, backfill và giám sát độ trễ.
- **Lệch phép chiếu:** HSD là geostationary projection; phải reproject trước khi chồng ranh giới hoặc ghép với IMERG.
- **Ngày/đêm:** các băng khả kiến không có tín hiệu phản xạ ban đêm; baseline nên ưu tiên IR/WV.
- **Sai khác thời gian quét:** một Full Disk cần khoảng 10 phút; cần lưu cả nominal time và observation time nếu mô hình nhạy với thời gian.
- **Ranh giới hành chính thay đổi:** phải khóa `boundary_version`, đặc biệt với Đà Nẵng sau sáp nhập 2025.
- **Điều khoản nguồn:** NOAA/JMA yêu cầu ghi nguồn và không được hàm ý cơ quan cung cấp chứng thực sản phẩm. JAXA P-Tree và NICT có điều kiện sử dụng/tái phân phối riêng, cần đọc lại trước khi công bố sản phẩm.
- **Sản phẩm L2 mây NOAA rất lớn:** kiểm tra ngày 2026-09-11 cho thấy ba file Full Disk CHGT/CMSK/CPHS của một slot tổng cộng trên 1 GiB. Không nên chọn làm luồng chính nếu chỉ cần ROI Việt Nam.

## 10. Tiêu chí PoC

PoC được xem là đạt khi:

- Thu đủ tối đa 48 frame/ngày ở cấu hình 30 phút, có log rõ các frame thiếu.
- Không có timestamp trùng; crawler chạy lại không tải/ghi trùng dữ liệu đã hoàn chỉnh.
- Latency p95 mục tiêu không quá 25 phút trong thời gian thử nghiệm.
- Sáu channel có cùng grid, shape, timestamp và mask.
- Bản đồ B13 khớp đường bờ Việt Nam và ranh giới tỉnh/thành sau reproject.
- Có hai view: toàn Việt Nam và Đà Nẵng; Đà Nẵng được xác định rõ theo boundary version.
- Tensor Himawari ghép được với đúng frame IMERG mà không nội suy thời gian.

## 11. Quyết định đề xuất

Chọn cấu hình baseline sau:

```yaml
source: noaa-himawari9
product: AHI-L1b-FLDK
cadence_minutes: 30
slots_utc: ["HH:00", "HH:30"]
bands: [B08, B09, B10, B13, B14, B15]
segments: [S03, S04, S05]
vietnam_bbox: [102.0, 8.0, 110.0, 24.5]
output_grid: match_existing_imerg_grid
raw_retention_days: 3
```

Chỉ chuyển sang nhịp 1 giờ nếu giới hạn lưu trữ/băng thông quan trọng hơn khả năng theo dõi sự phát triển nhanh của mây đối lưu.

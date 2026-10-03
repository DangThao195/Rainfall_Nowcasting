# Phân tích Đặc trưng (Feature Analysis) & Tiền xử lý dữ liệu

Trong bộ dữ liệu khí tượng ERA5 chúng ta đang sử dụng, mỗi "Feature" (đặc trưng) đại diện cho một biến số vật lý. Để mạng Neural học hiệu quả, chúng ta phải thấu hiểu bản chất phân phối (Distribution) của từng biến và từ đó đưa ra chiến lược xử lý phù hợp.

## 1. Cấu trúc các Feature
Dữ liệu đầu vào của chúng ta là một Tensor đa chiều, trong đó mỗi điểm ảnh (Pixel) chứa 5 biến số:
1. **Lượng mưa (`tp` - Total Precipitation):** Biến mục tiêu (Target Variable). Đo bằng đơn vị `mm`.
2. **Nhiệt độ 2m (`t2m`):** Nhiệt độ không khí bề mặt. Thường được quy đổi từ Kelvin sang độ C (Celsius).
3. **Áp suất mực nước biển trung bình (`msl` - Mean Sea Level Pressure):** Đo bằng `Pascal (Pa)`. Áp suất thấp thường báo hiệu thời tiết xấu, đối lưu mạnh (bão, áp thấp).
4. **Gió Đông - Tây (`u10`):** Vận tốc gió theo trục ngang (m/s). $u > 0$ là gió thổi từ Tây sang Đông.
5. **Gió Bắc - Nam (`v10`):** Vận tốc gió theo trục dọc (m/s). $v > 0$ là gió thổi từ Nam lên Bắc.

## 2. Đặc điểm Phân phối (Distribution)
Việc khảo sát phân phối của các biến rất quan trọng trong Machine Learning. 

### A. Nhiệt độ, Áp suất và Gió
- **Phân phối:** Các biến số tự nhiên này thường tuân theo **Phân phối chuẩn (Normal Distribution)** hoặc hình quả chuông (Bell-curve). 
- Nhiệt độ xoay quanh mức trung bình $25^\circ C - 30^\circ C$ ở Việt Nam, áp suất xoay quanh $1013 \text{ hPa}$.
- **Vấn đề lệch Scale:** Áp suất có giá trị lên tới $\approx 100,000$, trong khi Gió chỉ dao động từ $-10$ đến $10$. Nếu đưa nguyên gốc vào mạng Neural, Gradient sẽ bị bùng nổ (Exploding Gradients) hoặc mạng sẽ chỉ tập trung vào biến Áp suất vì số của nó quá to.

### B. Lượng mưa (Kẻ thù của Machine Learning)
- **Phân phối:** Lượng mưa có phân phối **Cực kỳ bất đối xứng (Highly Skewed / Zero-inflated)**.
- **Lý do:** Thời tiết Việt Nam có đến 80-90% thời gian là tạnh ráo (Lượng mưa = 0). Chỉ có khoảng 10-20% thời gian là có mưa, và số lượng những cơn mưa cực to (bão) lại vô cùng hiếm.
- **Vấn đề:** Nếu đưa vào mô hình AI thông thường, AI sẽ nhận ra rằng: *"Nếu mình cứ nhắm mắt đoán mưa = 0 liên tục, thì độ chính xác của mình vẫn lên tới 90%!"*. Từ đó sinh ra hiện tượng mô hình luôn luôn dự đoán "Không mưa" (All-zero prediction).

## 3. Chiến lược Tiền xử lý (Preprocessing Strategy)
Để giải quyết các vấn đề trên, trong file tiền xử lý (hoặc module DataLoader), chúng ta cần áp dụng:

### Kỹ thuật 1: Min-Max Scaling hoặc Z-Score Normalization
- Mục đích: Kéo tất cả các biến về cùng một dải giá trị từ $[0, 1]$ hoặc có trung bình = 0, phương sai = 1.
- Áp dụng cho: Tất cả các biến `t2m, msl, u10, v10`.
- Công thức: $X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$

### Kỹ thuật 2: Log-Transform (Biến đổi Logarit) cho Lượng mưa
- Mục đích: Giảm sự chênh lệch quá mức giữa cơn mưa nhỏ ($1mm$) và cơn mưa cực đoan ($100mm$), giúp mạng Neural không bị nhiễu.
- Áp dụng cho: `tp`
- Công thức: $P_{transformed} = \log(P + 1)$. (Cộng 1 để tránh lỗi $\log(0)$).

### Kỹ thuật 3: Masked Loss (Cắt tỉa Hàm suy hao)
- Trong hàm Loss của mạng, chúng ta nên nhân một trọng số lớn hơn (ví dụ x5 hoặc x10) cho các pixel **CÓ MƯA**, và phạt nhẹ tay với các pixel **KHÔNG MƯA**. Việc này giải quyết dứt điểm tính lười biếng (chỉ đoán $0$) của mạng Neural.

# ĐỀ XUẤT NGHIÊN CỨU & CẨM NANG HUẤN LUYỆN MÔ HÌNH HỌC SÂU DỰ BÁO LƯỢNG MƯA CỰC NGẮN (RAINFALL NOWCASTING) TRÊN BỘ DỮ LIỆU ĐA BIẾN ERA5

> **Tên đề tài:** Nghiên cứu và Phát triển Mô hình Học sâu Không gian - Thời gian (Spatiotemporal Deep Learning) kết hợp Ràng buộc Vật lý Khí quyển (Physics-Informed Neural Networks) phục vụ Dự báo Lượng mưa Cực ngắn (Nowcasting 0 - 3 giờ / 6 giờ) tại Lãnh thổ và Vùng biển Đảo Việt Nam.  
> **Dữ liệu nghiên cứu:** Bộ dữ liệu Khí tượng Tái phân tích Toàn cầu ECMWF ERA5 giai đoạn 2023 - 2025 (81x81 Grid, độ phân giải 0.25° ~ 27.75 km).  
> **Phạm vi không gian:** Vĩ độ $5.0^\circ N - 25.0^\circ N$, Kinh độ $100.0^\circ E - 120.0^\circ E$ (Bao phủ trọn vẹn 34 đơn vị hành chính tỉnh/thành phố của Việt Nam, Quần đảo Hoàng Sa, Quần đảo Trường Sa và Biển Đông).

---

## MỤC LỤC
1. [I. Tổng Quan Đề Tài & Tính Cấp Thiết](#i-tổng-quan-đề-tài--tính-cấp-thiết)
2. [II. Phân Tích Đặc Thù Bộ Dữ Liệu ERA5 & Thách Thức Vật Lý](#ii-phân-tích-đặc-thù-bộ-dữ-liệu-era5--thách-thức-vật-lý)
3. [III. Đề Xuất Các Hướng Kiến Trúc Mô Hình Học Sâu (Model Architectures)](#iii-đề-xuất-các-hướng-kiến-trúc-mô-hình-học-sâu-model-architectures)
4. [IV. Thiết Kế Hệ Thống Hàm Mất Mát Tối Ưu (Advanced Loss Functions)](#iv-thiết-kế-hệ-thống-hàm-mất-mát-tối-ưu-advanced-loss-functions)
5. [V. Hệ Thống Tiêu Chí Đánh Giá Chuẩn Khí Tượng Quốc Tế (Evaluation Benchmark)](#v-hệ-thống-tiêu-chí-đánh-giá-chuẩn-khí-tượng-quốc-tế-evaluation-benchmark)
6. [VI. Kế Hoạch Thực Nghiệm & Phân Tích Đóng Góp (Ablation Study)](#vi-kế-hoạch-thực-nghiệm--phân-tích-đóng-góp-ablation-study)
7. [VII. Chiến Lược Báo Cáo & Trả Lời Câu Hỏi Hội Đồng Coi Thi / Nghiệm Thu](#vii-chiến-lược-báo-cáo--trả-lời-câu-hỏi-hội-đồng-coi-thi--nghiệm-thu)

---

## I. TỔNG QUAN ĐỀ TÀI & TÍNH CẤP THIẾT

### 1.1. Bối cảnh Thực tiễn tại Việt Nam
- Việt Nam nằm trong vùng nhiệt đới gió mùa châu Á, là một trong những quốc gia chịu ảnh hưởng nặng nề nhất bởi các hiện tượng thời tiết cực đoan (bão nhiệt đới, áp thấp nhiệt đới, mưa dông đối lưu cực lớn, lũ quét, sạt lở đất miền Trung và ngập lụt đô thị tại Hà Nội, TP. Hồ Chí Minh, Đà Nẵng).
- **Dự báo lượng mưa cực ngắn (Nowcasting - từ 0 đến 3 giờ hoặc 6 giờ)** đóng vai trò quyết định trong việc cảnh báo khẩn cấp cho người dân, vận hành an toàn các hồ chứa thuỷ điện, giao thông đường thuỷ - đường không và ứng phó thiên tai tức thì.

### 1.2. Hạn chế của Các Phương pháp Truyền thống
- **Dự báo số trị thời tiết (NWP - Numerical Weather Prediction):** Độ chính xác cao ở hạn dài (1 - 5 ngày) nhưng thời gian tính toán mô phỏng quá lâu (mất vài giờ để giải hệ phương trình động lực học chất lưu), không thể đáp ứng yêu cầu tức thời của Nowcasting (dưới 5 phút).
- **Phép ngoại suy dòng quang học (Optical Flow / Radar Tracking):** Chỉ dựa trên chuyển động hình học thuần túy của mây mưa, giả định mây không sinh ra (Initiation) hoặc tiêu tan (Decay), dẫn đến sai số rất lớn sau 1 - 2 giờ.

### 1.3. Đột phá từ Phương pháp Tiếp cận Đa biến (Multivariate Spatiotemporal AI)
- Thay vì chỉ nhìn vào ảnh lượng mưa đơn lẻ trong quá khứ, mô hình học sâu của chúng ta tiếp nhận **5 trường biến số vật lý đồng thời**:
  1. $tp$ (Total Precipitation): Lượng mưa mặt đất (mm).
  2. $t_{2m}$ (2m Temperature): Nhiệt độ không khí bề mặt (°C) - Thể hiện năng lượng nhiệt và bất ổn định khí quyển.
  3. $msl$ (Mean Sea Level Pressure): Khí áp mực biển (hPa) - Thể hiện các tâm áp thấp, bão và rãnh hội tụ.
  4. $u_{10}, v_{10}$ (10m Wind Components): Vector hướng gió và vận tốc gió bề mặt (m/s) - Thể hiện cơ chế bình lưu (Advection) đẩy các khối mây di chuyển.

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               5 KÊNH ĐẦU VÀO ERA5 (t-5 -> t0)           │
                    │   tp (Mưa) | t2m (Nhiệt) | msl (Áp suất) | u10,v10 (Gió) │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                                                 ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │      MÔ HÌNH HỌC SÂU KHÔNG GIAN - THỜI GIAN (AI CORE)   │
                    │   (ConvLSTM / PredRNN / Earthformer / ThoR PINN)        │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                                                 ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │          DỰ BÁO LƯỢNG MƯA 6 FRAMES TƯƠNG LAI (t1 -> t6)  │
                    │          Bản đồ Lượng Mưa (mm) + Bản đồ Tỉnh Thành       │
                    │          + Bao bọc Quần đảo Hoàng Sa & Trường Sa        │
                    └─────────────────────────────────────────────────────────┘
```

---

## II. PHÂN TÍCH ĐẶC THÙ BỘ DỮ LIỆU ERA5 & THÁCH THỨC VẬT LÝ

### 2.1. Cấu trúc Lưới Không gian & Khẳng định Chủ quyền Lãnh thổ
- **Kích thước lưới:** $81 \times 81$ điểm ảnh (Grid resolution $0.25^\circ \approx 27.75\text{ km}$).
- **Toạ độ địa lý:** Vĩ độ $[5.0^\circ N, 25.0^\circ N]$, Kinh độ $[100.0^\circ E, 120.0^\circ E]$.
- **Ý nghĩa địa lý & Chính trị:** 
  - Toàn bộ 34 đơn vị hành chính của Việt Nam (năm 2026) được số hoá ranh giới chi tiết.
  - Vùng biển Đông rộng lớn với **Quần đảo Hoàng Sa (thuộc TP. Đà Nẵng)** tại $[15.5^\circ - 17.5^\circ N, 111.0^\circ - 113.2^\circ E]$ và **Quần đảo Trường Sa (thuộc tỉnh Khánh Hòa)** tại $[6.5^\circ - 12.5^\circ N, 111.5^\circ - 117.5^\circ E]$ được khoanh vùng bảo đảm tính toàn vẹn lãnh thổ và chủ quyền quốc gia trong đồ án nghiên cứu khoa học.

### 2.2. Bốn Thách thức Kỹ thuật Cốt lõi của Bộ Dữ liệu
1. **Hiện tượng Zero-Inflation (Mất cân bằng dữ liệu cực đoan):**
   - Kết quả EDA cho thấy: **88.4% điểm ảnh trong dataset có lượng mưa $< 0.1\text{ mm/h}$ (Tạnh ráo)**. Chỉ có **2.6% là mưa vừa** và **0.8% là mưa to đến rất to**.
   - *Hậu quả nếu xử lý sai:* Mạng Neural sẽ học ra dự đoán "Toàn số 0" (All-zero prediction) để giảm tối đa hàm suy hao MSE.
2. **Lệch bậc độ lớn giữa các biến số (Scale Disparity):**
   - Khí áp $msl \approx 100,000\text{ Pa}$, Nhiệt độ $t_{2m} \approx 300\text{ K}$, Gió $u, v \approx \pm 10\text{ m/s}$, Lượng mưa $tp \approx 0.001\text{ m}$.
   - Cần chuẩn hoá Min-Max riêng biệt cho từng kênh và áp dụng chuẩn hoá chỉ dựa trên tập Train để chống rò rỉ thông tin (Data Leakage).
3. **Hiện tượng Làm mịn / Nhòe ảnh (Blurring Effect) ở các bước dự báo xa ($t+4, t+5, t+6$):**
   - Các mô hình hồi quy truyền thống khi không chắc chắn về vị trí mưa to sẽ có xu hướng "trung bình hoá", dẫn đến các tâm mưa bão bị mờ nhạt, làm mất các đỉnh cực trị nguy hiểm.
4. **Tính bất định và đối lưu phi tuyến (Convective Non-linearities):**
   - Mưa nhiệt đới tại Việt Nam chịu chi phối mạnh bởi chu kỳ nhiệt ngày đêm (đỉnh mưa đối lưu 14h - 18h) và hoàn lưu gió mùa (Tây Nam và Đông Bắc).

---

## III. ĐỀ XUẤT CÁC HƯỚNG KIẾN TRÚC MÔ HÌNH HỌC SÂU (MODEL ARCHITECTURES)

Chúng tôi đề xuất 4 nhóm kiến trúc mô hình từ Baseline tiêu chuẩn đến State-Of-The-Art (SOTA) để so sánh đối chuẩn (Benchmarking):

```
                                    CÁC HƯỚNG MÔ HÌNH NGHIÊN CỨU
                                                  │
          ┌───────────────────────┬───────────────┴───────────────┬────────────────────────┐
          ▼                       ▼                               ▼                        ▼
  1. Spatiotemporal RNN    2. Transformer Models         3. Physics-Informed (PINN)  4. Generative / Diffusion
  - ConvLSTM (Baseline)    - Earthformer (Cuboid SOTA)   - ThoR PINN (TFC Constrained - Spatiotemporal Diffusion
  - PredRNN / PredRNN++    - Swin-UNet (Shifted Window)    + Advection PDE Loss)     - LDM Nowcast (Sharper Rain)
```

---

### Hướng 1: Mô hình Mạng Nơ-ron Hồi quy Không - Thời gian (Spatiotemporal RNNs)

#### 1. ConvLSTM (Shi et al., NIPS 2015) - Baseline Vững chắc
- **Nguyên lý:** Thay thế các phép nhân ma trận trọng số trong tế bào LSTM tiêu chuẩn bằng các toán tử tích chập không gian 2D (Convolutional Operators).
- **Công thức tế bào:**
  $$\begin{aligned}
  i_t &= \sigma(W_{xi} * \mathcal{X}_t + W_{hi} * \mathcal{H}_{t-1} + b_i) \\
  f_t &= \sigma(W_{xf} * \mathcal{X}_t + W_{hf} * \mathcal{H}_{t-1} + b_f) \\
  \mathcal{C}_t &= f_t \circ \mathcal{C}_{t-1} + i_t \circ \tanh(W_{xc} * \mathcal{X}_t + W_{hc} * \mathcal{H}_{t-1} + b_c) \\
  o_t &= \sigma(W_{xo} * \mathcal{X}_t + W_{ho} * \mathcal{H}_{t-1} + b_o) \\
  \mathcal{H}_t &= o_t \circ \tanh(\mathcal{C}_t)
  \end{aligned}$$
- **Ưu điểm:** Cấu trúc gọn nhẹ, huấn luyện nhanh, dễ triển khai, tiêu thụ ít VRAM.
- **Hạn chế:** Các trạng thái ẩn $\mathcal{H}_t$ bị ràng buộc trong từng tầng, khả năng truyền thông tin không gian đa tầng bị suy giảm khi dự báo dài hạn.

#### 2. PredRNN / PredRNN++ (Wang et al., TPAMI 2022) - Nâng cấp Bộ nhớ Đa tầng
- **Cải tiến:** Giới thiệu dòng chảy bộ nhớ Spatiotemporal Memory Flow $\mathcal{M}_t^l$ di chuyển theo hình ziczac (cả theo chiều thời gian ngang và chiều sâu kiến trúc dọc).
- **Bộ nhớ Gradient Highway:** Ngăn chặn hiện tượng triệt tiêu gradient qua các bước thời gian dài.

---

### Hướng 2: Mô hình Transformer Không gian Địa cầu (Earth System Transformers)

#### 1. Earthformer: Exploring Space-Time Cuboid Attention (Gao et al., NeurIPS 2022) - SOTA Khí tượng
- **Nguyên lý:** Chia tensor không gian - thời gian 3D $\mathbf{X} \in \mathbb{R}^{T \times H \times W \times C}$ thành các khối lập phương (Cuboids). Cơ chế **Cuboid Self-Attention** tính toán tương quan toàn cục hiệu quả với độ phức tạp tuyến tính thay vì bậc hai.
- **Ưu điểm vượt trội:**
  - Nắm bắt hoàn hảo các tương quan xa (Long-range teleconnections) - ví dụ: một vùng áp thấp ở Biển Đông cách xa 500 km gây mưa lớn ở đất liền miền Trung.
  - Giữ được độ sắc nét và cấu trúc dải mây đối lưu tốt hơn 20 - 30% so với ConvLSTM.

#### 2. Swin-UNet / SwinLSTM (Hierarchical Vision Transformer)
- Sử dụng cửa sổ trượt Shifted Windows (Swin) để trích xuất đặc trưng đa độ phân giải (Multi-scale feature extraction).

---

### Hướng 3: Mô hình Tích hợp Tri thức Vật lý (Physics-Informed Neural Networks - ThoR Framework)

Đây là điểm nhấn học thuật và sáng tạo cao nhất của đề tài, kết hợp toán học giải tích và cơ học chất lưu khí quyển:

```
                            ┌──────────────────────────────────────────────┐
                            │    Đầu vào X (5 biến ERA5: tp, t2m, msl, u, v) │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │    Spatiotemporal Backbone (ConvLSTM / UNet)  │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼ N(t) (Neural Residuals)
                            ┌──────────────────────────────────────────────┐
                            │      LỚP RÀNG BUỘC CỨNG TFC (t = 0 Exact)    │
                            │      P_hat(t) = P_0 + N(t) * (1 - e^(-t))    │
                            └──────────────────────┬───────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
            ┌─────────────────────────┐                         ┌─────────────────────────┐
            │   Data-driven Loss      │                         │  Physics Advection Loss │
            │   L_data = BMSE(P, Y)   │                         │  L_PDE = ||R_PDE||^2   │
            └────────────┬────────────┘                         └────────────┬────────────┘
                         │                                                   │
                         └─────────────────────────┬─────────────────────────┘
                                                   │
                                                   ▼
                                         L_total = L_data + λ * L_PDE
```

1. **Khối Nội suy Ràng buộc TFC (Theory of Functional Connections):**
   - Ép buộc dự báo tại thời điểm xuất phát $t=0$ luôn trùng khớp $100\%$ với ảnh lượng mưa quan sát thực tế $P_0$:
     $$\hat{P}(t) = P_0 + \mathbf{N}(t) \cdot \left(1 - e^{-t}\right)$$
   - *Ý nghĩa:* Loại bỏ hoàn toàn hiện tượng "nhảy bước" (Step-discontinuity) khi mô hình bắt đầu sinh dự báo.

2. **Hàm Phạt Đạo hàm Riêng Vật lý (Physics Advection-Diffusion PDE Loss):**
   - Dựa trên phương trình bảo toàn khối lượng và chuyển động bình lưu trong khí quyển:
     $$\mathcal{R}_{\text{PDE}} = \frac{\partial \hat{P}}{\partial t} + u_{10} \frac{\partial \hat{P}}{\partial x} + v_{10} \frac{\partial \hat{P}}{\partial y} - D \left(\frac{\partial^2 \hat{P}}{\partial x^2} + \frac{\partial^2 \hat{P}}{\partial y^2}\right)$$
   - Các đạo hàm không gian $\frac{\partial \hat{P}}{\partial x}, \frac{\partial \hat{P}}{\partial y}$ được tính toán tự động bằng tích chập 2D với **Bộ lọc Sobel cố định** trong PyTorch, cho phép gradient lan truyền ngược (Backpropagation) mượt mà mà không làm tăng chi phí tính toán.
   - *Ý nghĩa:* Mô hình bị phạt nặng nếu dự đoán mây mưa di chuyển ngược hướng gió thực tế $(u_{10}, v_{10})$.

---

### Hướng 4: Mô hình Sinh Khuếch tán Xác suất (Spatiotemporal Latent Diffusion Models)
- **Mục tiêu:** Khắc phục triệt để hiện tượng nhòe ảnh (Blurring effect) của mô hình hồi quy.
- Mô hình khuếch tán học phân phối xác suất có điều kiện $p(Y_{1:T} \mid X_{-T:0})$, sinh ra các mẫu dự báo có kết cấu mưa tự nhiên và bảo toàn phương sai không gian.

---

## IV. THIẾT KẾ HỆ THỐNG HÀM MẤT MÁT TỐI ƯU (ADVANCED LOSS FUNCTIONS)

Để giải quyết triệt để tính chất Zero-Inflation và duy trì độ sắc nét của vùng mưa cực đoan, chúng tôi thiết kế hàm mất mát tổng hợp:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BMSE}} + \alpha \mathcal{L}_{\text{BMAE}} + \beta \mathcal{L}_{\text{PDE}} + \gamma \mathcal{L}_{\text{Freq}}$$

### 4.1. Balanced Mean Squared Error (B-MSE) & Balanced MAE (B-MAE)
Nhân trọng số phạt tăng dần theo cấp độ mưa thực tế:
$$w(y) = \begin{cases} 
1.0, & y < 0.1\text{ mm/h (Không mưa)} \\
2.0, & 0.1 \le y < 2.0\text{ mm/h (Mưa nhỏ)} \\
5.0, & 2.0 \le y < 10.0\text{ mm/h (Mưa vừa)} \\
10.0, & 10.0 \le y < 25.0\text{ mm/h (Mưa to)} \\
20.0, & y \ge 25.0\text{ mm/h (Mưa rất to / Bão)}
\end{cases}$$

$$\mathcal{L}_{\text{BMSE}} = \frac{1}{B \cdot T \cdot H \cdot W} \sum_{b, t, i, j} w(y_{b,t,i,j}) \cdot \left(\hat{y}_{b,t,i,j} - y_{b,t,i,j}\right)^2$$

### 4.2. Focal Frequency Loss (FFL) - Chống Mờ Vùng Tâm Bão
Sử dụng biến đổi Fourier 2D rời rạc (2D Discrete Fourier Transform) để ép mô hình học đúng các thành phần tần số cao (High-frequency details) - chính là các biên cạnh sắc nét của tâm bão.

---

## V. HỆ THỐNG TIÊU CHÍ ĐÁNH GIÁ CHUẨN KHÍ TƯỢNG QUỐC TẾ (EVALUATION BENCHMARK)

Đánh giá mô hình trên tập kiểm thử độc lập **Test Set (Toàn bộ năm 2025)** qua 2 nhóm tiêu chí:

### 5.1. Nhóm Chỉ số Sai số Liên tục (Continuous Metrics)
1. **Root Mean Squared Error (RMSE):** Đo độ lệch bình phương trung bình:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$
2. **Mean Absolute Error (MAE):** Đo độ lệch tuyệt đối trung bình:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$
3. **Correlation Coefficient (CC / Pearson $r$):** Đo mức độ tương đồng về hình thái không gian giữa dự báo và thực tế.

---

### 5.2. Nhóm Chỉ số Phân loại Khí tượng (Categorical Skill Scores)
Xác định ma trận phân loại dựa trên 4 ngưỡng mưa quan trọng:
- Ngưỡng 1: $r \ge 0.1\text{ mm/h}$ (Có mưa vs Không mưa)
- Ngưỡng 2: $r \ge 1.0\text{ mm/h}$ (Mưa nhỏ)
- Ngưỡng 3: $r \ge 5.0\text{ mm/h}$ (Mưa vừa)
- Ngưỡng 4: $r \ge 10.0\text{ mm/h}$ (Mưa to)
- Ngưỡng 5: $r \ge 25.0\text{ mm/h}$ (Mưa rất to / Bão)

| Dự báo \ Thực tế | Mưa ($y \ge \tau$) | Không mưa ($y < \tau$) |
| :--- | :---: | :---: |
| **Dự báo có mưa ($\hat{y} \ge \tau$)** | **Hits ($H$)** (Đoán đúng) | **False Alarms ($F$)** (Báo động giả) |
| **Dự báo không mưa ($\hat{y} < \tau$)** | **Misses ($M$)** (Bỏ sót) | **Correct Rejections ($R$)** (Đúng tạnh) |

1. **Critical Success Index (CSI / Threat Score) [Càng cao càng tốt, Max = 1.0]:**
   $$\text{CSI} = \frac{H}{H + M + F}$$
2. **Probability of Detection (POD / Hit Rate) [Càng cao càng tốt, Max = 1.0]:**
   $$\text{POD} = \frac{H}{H + M}$$
3. **False Alarm Ratio (FAR) [Càng thấp càng tốt, Min = 0.0]:**
   $$\text{FAR} = \frac{F}{H + F}$$
4. **Heidke Skill Score (HSS) [Càng cao càng tốt, Max = 1.0]:** Đo lường độ chính xác đã loại trừ yếu tố may rủi ngẫu nhiên.

---

## VI. KẾ HOẠCH THỰC NGHIỆM & PHÂN TÍCH ĐÓNG GÓP (ABLATION STUDY)

### 6.1. Bảng Kế hoạch Thí nghiệm Đề xuất

| Mã Thí Nghiệm | Mô hình (Architecture) | Đầu vào (Input Variables) | Hàm Loss (Loss Function) | Mục tiêu Kiểm chứng |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-01 (Baseline 1)** | Persistence / Optical Flow | Chỉ lượng mưa $tp$ | N/A | Chuẩn đối sánh cổ điển |
| **EXP-02 (Baseline 2)** | ConvLSTM (Đơn biến) | Chỉ lượng mưa $tp$ | MSE tiêu chuẩn | Hiệu năng mô hình Deep Learning đơn biến |
| **EXP-03 (Proposed 1)** | ConvLSTM (Đa biến) | $tp, t_{2m}, msl, u_{10}, v_{10}$ | MSE tiêu chuẩn | Chứng minh giá trị của tiếp cận Đa biến |
| **EXP-04 (Proposed 2)** | ConvLSTM (Đa biến) | $tp, t_{2m}, msl, u_{10}, v_{10}$ | Balanced MSE | Chứng minh hiệu quả giải quyết Zero-Inflation |
| **EXP-05 (Proposed 3)** | ThoR-PINN (TFC + Advection) | $tp, t_{2m}, msl, u_{10}, v_{10}$ | $\text{BMSE} + \lambda \mathcal{L}_{\text{PDE}}$ | Chứng minh vai trò của Ràng buộc Vật lý Khí quyển |
| **EXP-06 (SOTA)** | Earthformer (Cuboid Attn) | $tp, t_{2m}, msl, u_{10}, v_{10}$ | $\text{BMSE} + \text{FFL}$ | Kiểm chứng kiến trúc Transformer hiện đại nhất |

### 6.2. Các Bài Phân Tích Đóng Góp (Ablation Studies)
1. **Ablation trên từng biến đầu vào:** Thử nghiệm ngắt bỏ lần lượt từng kênh ($msl$, $t2m$, $u10/v10$) để xem biến nào đóng góp nhiều nhất vào việc giảm sai số dự báo.
2. **Ablation trên Lead Time (Thời gian dự báo trước):** Đánh giá tốc độ suy giảm chỉ số CSI từ $t+1$ (30 phút / 1 giờ) đến $t+6$ (3 giờ / 6 giờ).
3. **Phân tích Đất liền vs Biển Đảo:** Đánh giá độ chính xác riêng cho 34 đơn vị hành chính đất liền và 2 quần đảo Hoàng Sa - Trường Sa.

---

## VII. CHIẾN LƯỢC BÁO CÁO & TRẢ LỜI CÂU HỎI HỘI ĐỒNG COI THI / NGHIỆM THU

Dưới đây là cẩm nang giúp bạn tự tin đạt điểm tuyệt đối trước Hội đồng chấm đề tài:

### 7.1. Cách Trình bày Slide & Điểm Nhấn Thị Giác (Visual Storytelling)
1. **Slide 1 - Bản đồ Chủ quyền & Tổng quan Dữ liệu:**
   - Chiếu hình ảnh bản đồ từ file `img/01_multivariable_spatial_map.png` và `img/sample_vietnam_map.png`.
   - *Lời nói mẫu:* *"Kính thưa Hội đồng, đề tài của em sử dụng dữ liệu tái phân tích ERA5 trên toàn bộ không gian lãnh thổ và vùng biển đảo Việt Nam, bao gồm đầy đủ ranh giới 34 đơn vị hành chính và hai quần đảo thiêng liêng Hoàng Sa, Trường Sa được khoanh vùng kiểm soát ranh giới rõ ràng."*
2. **Slide 2 - Hiện tượng Khí động học & Hoàn lưu Gió:**
   - Chiếu hình ảnh `img/02_wind_streamlines_rain_advection.png`.
   - *Lời nói mẫu:* *"Qua phân tích khí động học, ta thấy rõ các tâm mây mưa di chuyển hoàn toàn ăn khớp theo các đường dòng của vector gió 10m. Đây chính là cơ sở khoa học để nhóm em tích hợp phương trình bình lưu vào hàm Loss của mạng PINN."*
3. **Slide 3 - Vấn đề Zero-Inflation & Giải pháp Kỹ thuật:**
   - Chiếu hình ảnh `img/03_rain_distribution_and_categories.png`.
   - *Lời nói mẫu:* *"88.4% dữ liệu là không mưa. Nếu dùng MSE thông thường, AI sẽ bị 'lười'. Nhóm em đã giải quyết bằng Balanced Sampler trong DataLoader và hàm Balanced MSE có trọng số."*
4. **Slide 4 - Chuỗi Diễn biến Bão Cực đoan & Đánh giá Dự báo:**
   - Chiếu hình ảnh `img/08_extreme_storm_sequence.png` và `img/11_evaluation_comparison_map.png`.

---

### 7.2. Top 5 Câu Hỏi Hóc Búa Hội Đồng Thường Hỏi & Câu Trả Lời Mẫu

#### Câu 1: *"Tại sao em không dùng luôn mô hình dự báo thời tiết số trị NWP (như WRF) của Trung tâm Khí tượng mà lại dùng AI Nowcasting?"*
- **Trả lời mẫu:** *"Dạ thưa Thầy/Cô, mô hình số trị NWP (như WRF) giải hệ phương trình vi phân phi tuyến Navier-Stokes rất chính xác cho dự báo trung hạn (1-3 ngày), nhưng chu kỳ chạy (Run-time) mất từ 1 đến 3 tiếng đồng hồ trên siêu máy tính. Trong khi đó, bài toán Nowcasting đòi hỏi phải đưa ra cảnh báo dông lốc, bão lũ trong vòng 1-5 phút ngay sau khi nhận dữ liệu. Mô hình AI sau khi huấn luyện xong chỉ mất chưa đầy 0.1 giây để suy luận trên GPU, đáp ứng tính chất cứu hộ khẩn cấp theo thời gian thực."*

#### Câu 2: *"Độ phân giải 0.25 độ (~28 km) của ERA5 có quá thô cho dự báo mưa đô thị không?"*
- **Trả lời mẫu:** *"Dạ thưa Thầy/Cô, ERA5 là bộ dữ liệu chuẩn mực toàn cầu cung cấp đầy đủ các trường nhiệt động lực học ($u, v, t, p$) mà radar hay đo mưa mặt đất không có được. Đề tài của em xây dựng khung kiến trúc nền tảng (Foundation Pipeline) trên ERA5. Toàn bộ kiến trúc ConvLSTM, ThoR-PINN và Data Pipeline này có tính kế thừa 100%, sẵn sàng nạp dữ liệu Radar phản xạ (độ phân giải 1 km) hoặc Vệ tinh GPM IMERG (10 km) khi triển khai thực tế."*

#### Câu 3: *"Tại sao mô hình PINN (Physics-Informed) lại vượt trội hơn mô hình Deep Learning thông thường?"*
- **Trả lời mẫu:** *"Dạ thưa Thầy/Cô, mô hình AI thuần túy (Black-box) chỉ học sự tương quan thống kê, nên khi gặp các cơn bão lịch sử chưa từng xuất hiện trong tập Train, AI sẽ dự báo sai lệch phi lý (ví dụ: mây mưa di chuyển ngược chiều gió hoặc tự nhiên tan biến vô cớ). Mô hình PINN của nhóm em bổ sung hàm phạt vật lý Advection Loss và ràng buộc TFC, bắt buộc mạng Neural phải tuân thủ định luật bảo toàn khối lượng và chuyển động dòng khí, giúp dự báo đáng tin cậy và có cơ sở khoa học."*

#### Câu 4: *"Em giải quyết hiện tượng mờ ảnh (Blurring Effect) khi dự báo các bước thời gian xa (t+4, t+5, t+6) như thế nào?"*
- **Trả lời mẫu:** *"Dạ thưa Thầy/Cô, hiện tượng mờ ảnh xảy ra do hàm mất mát MSE tối ưu theo giá trị kỳ vọng trung bình. Nhóm em đã áp dụng 3 giải pháp: (1) Thay thế MSE bằng Balanced MAE và Focal Frequency Loss (FFL) để phạt sai số tần số cao; (2) Tích hợp kiến trúc Earthformer với cơ chế Cuboid Attention giữ sắc nét vùng biên; và (3) Đề xuất hướng phát triển mô hình Diffusion để sinh mẫu xác suất bảo toàn phương sai mây mưa."*

#### Câu 5: *"Tập Test của em được chọn như thế nào để đảm bảo không bị Data Leakage?"*
- **Trả lời mẫu:** *"Dạ thưa Thầy/Cô, nhóm em phân chia dữ liệu nghiêm ngặt theo trục thời gian (Temporal Split): Toàn bộ năm 2023 đến giữa 2024 làm tập Train (18 tháng), nửa cuối 2024 làm tập Validation (6 tháng để chọn checkpoint tốt nhất), và trọn vẹn năm 2025 (12 tháng) làm tập Test độc lập. Toàn bộ tham số chuẩn hoá Min, Max, Mean, Std chỉ được tính toán trên tập Train và áp dụng cố định sang tập Test, tuyệt đối không có hiện tượng rò rỉ dữ liệu."*

---

## VIII. TỔNG KẾT

Bộ mã nguồn và tài liệu trong đề tài này đã hoàn thiện đầy đủ chuỗi giá trị:
1. `01_Download_ERA5.ipynb`: Tải dữ liệu tự động từ Copernicus CDS.
2. `02_EDA_ERA5.ipynb`: Khảo sát đa biến, bản đồ 34 đơn vị hành chính, khoanh vùng Hoàng Sa - Trường Sa, phân tích chu kỳ ngày đêm và phân phối mưa.
3. `03_Data_Preprocessing.ipynb`: Chuẩn hóa dữ liệu, chống rò rỉ, tạo PyTorch DataLoader tối ưu bộ nhớ.
4. `04_Model_Training_ConvLSTM.ipynb` & `04_Model_Training_PINN.ipynb`: Huấn luyện mô hình Spatiotemporal AI và mô hình Vật lý Khí quyển.
5. `05_Evaluation.ipynb`: Đánh giá chỉ số RMSE, MAE, CSI, POD, FAR, HSS và trực quan hoá dự báo.
6. `RESEARCH_PROPOSAL_AND_MODELING_GUIDE.md`: Bản đề xuất khoa học và cẩm nang bảo vệ luận văn hoàn chỉnh.

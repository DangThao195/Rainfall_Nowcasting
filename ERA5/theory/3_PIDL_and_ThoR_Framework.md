# Tổng hợp Khung PIDL ThoR cho Rainfall Nowcasting

Tài liệu này đóng vai trò kết nối lý thuyết Vật lý (Phương trình Bình lưu - Khuếch tán) và Toán học (Theory of Functional Connections) vào một vòng lặp huấn luyện (Training Loop) hoàn chỉnh của mạng học sâu (Deep Learning). Đây chính là phần hồn của luận văn: Hiện thực hóa lý thuyết thành các dòng code tối ưu.

## 1. Cấu trúc mạng (Architecture Overview)
Mô hình ThoR-PIDL bao gồm 3 khối kiến trúc chính chạy nối tiếp nhau:

1. **Khối trích xuất không gian - thời gian (Spatiotemporal Encoder-Decoder):**
   - Sử dụng các mạng như ConvLSTM hoặc U-Net.
   - Đầu vào $\mathbf{X}$: Chuỗi ảnh Radar/Vệ tinh lượng mưa các thời điểm quá khứ $t_{-k}, \dots, t_0$, cùng với các trường biến khí tượng đi kèm (Gió $u, v$, Nhiệt độ $T$, Áp suất $P$).
   - Đầu ra $\mathbf{N}$: Một xấp xỉ phần dư lượng mưa (Neural Residuals) chưa chịu bất kỳ ràng buộc nào.

2. **Khối Nội suy Ràng buộc (TFC Layer):**
   - Đầu vào: Đầu ra $\mathbf{N}$ của mạng Neural và Bản đồ lượng mưa ở thời điểm cuối cùng của hiện tại $P_0$.
   - Phép toán: Căng hàm TFC $\hat{P}(t) = P_0 + \mathbf{N}(t) \cdot (1 - e^{-t})$.
   - Đầu ra $\hat{P}$: Bản đồ dự báo lượng mưa cuối cùng, đảm bảo chính xác về mặt vật lý ở $t=0$.

3. **Khối Hàm Suy Hao Vật Lý (Physics-Informed Loss Module):**
   - Một hàm vi phân tự động (Automatic Differentiation) hoặc Vi phân rời rạc (Finite Differences) để tính sai số phương trình đạo hàm riêng.

## 2. Xây dựng Hàm Loss Tổng Hợp (The Loss Function Formulation)
Toàn bộ mô hình được tối ưu hóa dựa trên việc cực tiểu hóa hàm Loss tổng hợp:
$$ \mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda \mathcal{L}_{PDE} $$

### A. Data-driven Loss ($\mathcal{L}_{MSE}$)
Đây là hàm suy hao truyền thống trong Machine Learning, đo lường sự chênh lệch giữa dự đoán và thực tế:
$$ \mathcal{L}_{MSE} = \frac{1}{N_{train}} \sum_{i=1}^{N_{train}} \left| \hat{P}_i - P_{true, i} \right|^2 $$

### B. Physics-driven Loss ($\mathcal{L}_{PDE}$)
Mô hình bị phạt nếu lượng mưa không tuân thủ định luật bảo toàn khối lượng và cơ học chất lưu. Thay vì giải phương trình vi phân liên tục, ta rời rạc hóa nó trên ma trận điểm ảnh (Pixel Grid).

Sử dụng phép xấp xỉ sai phân hữu hạn (hoặc Convolution với bộ lọc Sobel):
- **Đạo hàm không gian:** 
  - $\frac{\partial \hat{P}}{\partial x} \approx \text{Conv2D}(\hat{P}, K_x)$
  - $\frac{\partial \hat{P}}{\partial y} \approx \text{Conv2D}(\hat{P}, K_y)$
  - (Với $K_x, K_y$ là ma trận kernel Sobel).
- **Đạo hàm thời gian:**
  - $\frac{\partial \hat{P}}{\partial t} \approx \frac{\hat{P}_{t+1} - \hat{P}_t}{\Delta t}$

Khi đó, thặng dư vật lý (Physics Residual) tại mỗi pixel được tính là:
$$ \mathcal{R}_{PDE} = \frac{\partial \hat{P}}{\partial t} + u \frac{\partial \hat{P}}{\partial x} + v \frac{\partial \hat{P}}{\partial y} - D \left( \frac{\partial^2 \hat{P}}{\partial x^2} + \frac{\partial^2 \hat{P}}{\partial y^2} \right) $$

Hàm phạt vật lý là trung bình bình phương của thặng dư này trên toàn bộ không gian và thời gian:
$$ \mathcal{L}_{PDE} = \frac{1}{N_{domain}} \sum_{i \in Domain} \left| \mathcal{R}_{PDE}^{(i)} \right|^2 $$

*(Lưu ý: Vector gió $u, v$ không do mạng tự học mà được **truyền trực tiếp từ dữ liệu ERA5 (`u10, v10`)** vào hàm Loss. Đây chính là yếu tố làm nên tên gọi "Motion-Dependent").*

## 3. Quá trình Lan truyền ngược (Backpropagation)
Khi PyTorch tính toán `loss.backward()`, luồng gradient sẽ chạy ngược từ cả 2 nhánh:
- Nhánh Data: Ép mạng đổi weights để khớp với nhãn.
- Nhánh PDE: Ép mạng đổi weights để tuân thủ phương trình dòng chảy.
Việc tính toán đạo hàm không gian bằng các phép chập `Conv2D` tĩnh (fixed weights) cho phép đồ thị tính toán (computational graph) của PyTorch lan truyền đạo hàm qua nó một cách trơn tru, không cản trở việc cập nhật tham số của mạng ConvLSTM lõi.

## Kết luận
Bằng việc kết hợp TFC (xử lý triệt để điều kiện ban đầu) và PIDL (tích hợp động lực học khí quyển vào không gian Loss), khung kiến trúc ThoR vượt qua những rào cản của AI đen (Black-box), tạo ra các dự báo thời tiết dài hạn (Lead times) với độ chính xác và tính hợp lý về mặt khí tượng học (Meteorologically sound) vượt trội.

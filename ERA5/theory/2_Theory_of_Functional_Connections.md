# Ràng buộc không gian nghiệm: Theory of Functional Connections (TFC)

Mạng Neural nhân tạo truyền thống (Unconstrained Neural Networks) là các hàm xấp xỉ phổ quát (Universal Approximators). Tuy nhiên, khi giải quyết các bài toán vật lý, mạng Neural thường vi phạm các điều kiện ban đầu (Initial Conditions - IC) và điều kiện biên (Boundary Conditions - BC) ở giai đoạn đầu của quá trình huấn luyện, dẫn đến hội tụ chậm hoặc rơi vào cực tiểu cục bộ (local minima).

**Theory of Functional Connections (TFC)** là một khung toán học biến đổi bài toán tối ưu hóa có ràng buộc (Constrained Optimization) thành bài toán tối ưu hóa không ràng buộc (Unconstrained Optimization) bằng cách **nội suy giải tích** các điều kiện biên.

## 1. Hạn chế của Soft-Constraints (Ràng buộc mềm)
Trong Physics-Informed Neural Networks (PINN) cơ bản, các điều kiện biên và ban đầu thường được đưa vào hàm Loss dưới dạng các số hạng phạt (Penalty terms):
$$ \mathcal{L} = \mathcal{L}_{PDE} + \lambda_{IC} \mathcal{L}_{IC} + \lambda_{BC} \mathcal{L}_{BC} $$
- Hạn chế: Việc dò tìm các siêu tham số $\lambda$ rất phức tạp. Hơn nữa, $\mathcal{L}_{IC}$ hiếm khi hội tụ chính xác tuyệt đối về $0$. Nghĩa là mô hình có thể xấp xỉ sai cả trạng thái thời tiết ở hiện tại ($t=0$).

## 2. Giải pháp Hard-Constraints bằng TFC
TFC tạo ra một **hàm biểu diễn có giới hạn (Constrained Expression)** $f(x, t)$ đảm bảo luôn luôn thỏa mãn chính xác tuyệt đối các ràng buộc, bất kể mạng Neural bên dưới có trọng số (weights) như thế nào.

Dạng tổng quát của một biểu thức TFC:
$$ f(x, t) = g(x, t) + \mathcal{N}(x, t) \cdot h(x, t) $$
Trong đó:
- $f(x, t)$: Output cuối cùng của mô hình.
- $g(x, t)$: Một hàm chiếu (Projection function) được thiết kế giải tích để thỏa mãn điều kiện biên/ban đầu.
- $\mathcal{N}(x, t)$: Mạng Neural tự do (ví dụ: ConvLSTM, U-Net). Output của nó có thể là bất kỳ giá trị nào.
- $h(x, t)$: Hàm triệt tiêu (Switching function). Hàm này có đặc tính: bằng $0$ tại các ranh giới (điều kiện biên) và khác $0$ ở không gian bên trong.

## 3. Ứng dụng TFC vào Bài toán Rainfall Nowcasting
Trong bài toán Nowcasting, ràng buộc quan trọng nhất là **Điều kiện ban đầu (Initial Condition)**. Nghĩa là tại thời điểm $t=0$, lượng mưa dự báo phải bằng chính xác lượng mưa thực tế đầu vào.

Giả sử $P(x, y, 0) = P_0(x, y)$ là bản đồ lượng mưa hiện tại. Ta thiết kế TFC như sau:
$$ \hat{P}(x, y, t) = P_0(x, y) + \mathcal{N}(x, y, t) \times (1 - e^{-\tau t}) $$

### Phân tích cơ chế hoạt động:
1. **Tại thời điểm $t=0$:** 
   - Số hạng $(1 - e^0) = 0$.
   - Do đó, phần tử $\mathcal{N}(x, y, t)$ bị triệt tiêu hoàn toàn.
   - Kết quả: $\hat{P}(x, y, 0) = P_0(x, y)$. **Ràng buộc được thỏa mãn chính xác 100% bằng toán học.**

2. **Tại các thời điểm $t > 0$ (Dự báo tương lai):**
   - Số hạng $(1 - e^{-\tau t})$ tăng dần về $1$.
   - Mạng lưới Neural $\mathcal{N}(x, y, t)$ bắt đầu phát huy tác dụng, dự đoán sự sai khác (residual) của lượng mưa so với trạng thái ban đầu.
   - Tham số $\tau$ kiểm soát tốc độ mạng lưới thoát khỏi sự ảnh hưởng của trạng thái ban đầu.

### Ý nghĩa thực tiễn trong Khung ThoR:
Việc sử dụng TFC (Constraint-Centric) mang lại 2 lợi ích khổng lồ:
1. **Giảm không gian tìm kiếm (Search space reduction):** Mạng Neural không cần lãng phí chu kỳ huấn luyện (Epochs) để học lại bản đồ lượng mưa ban đầu. Nó tập trung toàn bộ năng lực tính toán để học sự biến thiên động lực học trong tương lai.
2. **Loại bỏ sự cạnh tranh Gradient (Gradient Pathology):** Không còn sự xung đột giữa gradient của $\mathcal{L}_{IC}$ và $\mathcal{L}_{PDE}$ trong hàm Loss, giúp mô hình hội tụ ổn định hơn.

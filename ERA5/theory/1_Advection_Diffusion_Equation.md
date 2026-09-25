# Cơ sở Động lực học Khí quyển: Phương trình Bình lưu - Khuếch tán

Việc tích hợp Vật lý vào Học sâu (Physics-Informed Deep Learning - PIDL) trong bài toán dự báo lượng mưa đòi hỏi một nền tảng toán học vững chắc dựa trên cơ học chất lưu và nhiệt động lực học khí quyển. Phương trình cốt lõi chi phối sự dịch chuyển của các trường vô hướng (như lượng mưa, độ ẩm, nhiệt độ) trong khí quyển là **Phương trình Bình lưu - Khuếch tán (Advection-Diffusion Equation)**.

## 1. Định luật Bảo toàn Khối lượng (Continuity Equation)
Mọi mô hình vật lý khí quyển đều bắt nguồn từ định luật bảo toàn khối lượng. Trong một không gian kiểm soát thể tích $V$, sự biến thiên của mật độ vật chất $\rho$ theo thời gian phải bằng tổng thông lượng (flux) đi qua biên của thể tích đó cộng với nguồn sinh/diệt bên trong.

Phương trình bảo toàn tổng quát (Dạng vi phân):
$$ \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = S $$

Trong bài toán lượng mưa, ta xét trường vô hướng $P(x, y, t)$ biểu diễn lượng mưa (Precipitation) tại tọa độ không gian 2D $(x, y)$ và thời gian $t$. Vector vận tốc gió được biểu diễn là $\mathbf{v} = (u, v)$, tương ứng với thành phần gió Đông-Tây (`u10`) và Bắc-Nam (`v10`).

## 2. Thành phần Bình lưu (Advection Term)
Bình lưu mô tả sự vận chuyển của đại lượng $P$ do trường vận tốc vĩ mô $\mathbf{v}$ (Gió). Áp dụng quy tắc đạo hàm của tích:
$$ \nabla \cdot (P \mathbf{v}) = \mathbf{v} \cdot \nabla P + P (\nabla \cdot \mathbf{v}) $$

Nếu giả sử dòng không khí ở tầng thấp là dòng không chịu nén (Incompressible flow), phân kỳ của trường vận tốc bằng không ($\nabla \cdot \mathbf{v} = 0$). Khi đó:
$$ \nabla \cdot (P \mathbf{v}) \approx \mathbf{v} \cdot \nabla P = u \frac{\partial P}{\partial x} + v \frac{\partial P}{\partial y} $$

Thành phần này cực kỳ quan trọng trong *Motion-Dependent Framework*. Nó liên kết trực tiếp sự dịch chuyển của vùng mưa với hướng và tốc độ gió.

## 3. Thành phần Khuếch tán (Diffusion Term)
Ngoài chuyển động vĩ mô do gió, lượng mưa còn lan truyền do các xoáy rối vi mô (Turbulent diffusion) trong khí quyển. Dựa trên định luật Fick, thông lượng khuếch tán tỷ lệ thuận với gradient của nồng độ:
$$ \mathbf{J}_{diff} = -D \nabla P $$
Trong đó $D$ là hệ số khuếch tán. Đưa vào phương trình bảo toàn, ta có:
$$ \nabla \cdot \mathbf{J}_{diff} = -D \nabla^2 P $$

## 4. Thành phần Nguồn sinh/diệt (Source/Sink Term)
Lượng mưa không chỉ dịch chuyển mà còn được sinh ra hoặc mất đi do quá trình nhiệt động lực học (bốc hơi, ngưng tụ). Ta ký hiệu thành phần này là $S(x, y, t)$. $S$ phụ thuộc phức tạp vào sự chênh lệch Nhiệt độ ($T$) và Áp suất bề mặt ($MSL$). Trong kiến trúc PIDL, phần dư (residual) của mạng Neural thường đóng vai trò tự xấp xỉ thành phần phi tuyến này.

## 5. Tổng hợp Phương trình Đạo hàm riêng (PDE)
Kết hợp (1), (2), (3) và (4), phương trình chi phối lượng mưa trong không gian 2D được viết là:
$$ \frac{\partial P}{\partial t} + u \frac{\partial P}{\partial x} + v \frac{\partial P}{\partial y} = D \left( \frac{\partial^2 P}{\partial x^2} + \frac{\partial^2 P}{\partial y^2} \right) + S $$

### Diễn giải các số hạng:
1. $\frac{\partial P}{\partial t}$: Tốc độ biến thiên lượng mưa tại một điểm cố định.
2. $u \frac{\partial P}{\partial x} + v \frac{\partial P}{\partial y}$: Gradient lượng mưa bị dịch chuyển bởi vector gió (Advection).
3. $D \nabla^2 P$: Sự tơi ra/lan tỏa của đám mây do khuếch tán rối (Diffusion).
4. $S$: Nguồn sinh mây từ đối lưu nhiệt - áp.

**Trong khung ThoR (PIDL):** AI sẽ phải tìm ra một hàm $P(x, y, t)$ (tức là dự báo lượng mưa) sao cho biểu thức trên xấp xỉ bằng $0$. Nếu mạng dự báo một đám mây đi ngược lại với vector gió $(u, v)$, số hạng bình lưu sẽ tạo ra một sai số cực lớn, dẫn đến hàm Loss bị đẩy lên cao, ép thuật toán tối ưu phải điều chỉnh lại.

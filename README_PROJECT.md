# GDSearch - So sánh Thuật toán Tối ưu hóa

Dự án Python chuyên nghiệp để so sánh các thuật toán tối ưu hóa (SGD, SGD Momentum, RMSProp, Adam) trên các hàm kiểm tra 2D.

## 📋 Mô tả

Dự án này triển khai và so sánh hiệu suất của các thuật toán gradient descent khác nhau trên các hàm kiểm tra tối ưu hóa cổ điển:
- **Rosenbrock**: Hàm có thung lũng hẹp, khó tối ưu hóa
- **Ill-Conditioned Quadratic**: Hàm bậc hai với số điều kiện cao
- **Saddle Point**: Hàm có điểm yên ngựa

## 🗂️ Cấu trúc Dự án

```
GDSearch/
├── test_functions.py      # Định nghĩa các hàm kiểm tra
├── optimizers.py          # Triển khai các thuật toán tối ưu
├── run_experiment.py      # Script chạy thí nghiệm
├── plot_results.py        # Script trực quan hóa kết quả
├── requirements.txt       # Các thư viện phụ thuộc
├── results/              # Thư mục chứa kết quả CSV
├── plots/                # Thư mục chứa biểu đồ
└── README.md             # File này
```

## 🚀 Cài đặt

### 1. Clone hoặc tải dự án

```bash
cd /workspaces/GDSearch
```

### 2. Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

Các thư viện cần thiết:
- `numpy`: Tính toán số học
- `matplotlib`: Vẽ đồ thị
- `pandas`: Xử lý dữ liệu
- `tqdm`: Hiển thị thanh tiến trình

## 📊 Sử dụng

### Bước 1: Chạy Thí nghiệm

Chạy tất cả các thí nghiệm so sánh:

```bash
python run_experiment.py
```

Script này sẽ:
- Chạy 4 thuật toán tối ưu (SGD, SGD Momentum, RMSProp, Adam)
- Trên 3 hàm kiểm tra khác nhau
- Với 2 learning rates khác nhau (0.01, 0.001)
- Với 3 seed khác nhau để đảm bảo tính ổn định
- Tổng cộng: **72 thí nghiệm**

Kết quả được lưu trong thư mục `results/` dưới dạng file CSV.

### Bước 2: Trực quan hóa Kết quả

Tạo các biểu đồ từ kết quả thí nghiệm:

```bash
python plot_results.py
```

Script này sẽ tạo:
- **Biểu đồ quỹ đạo**: Hiển thị đường đi của thuật toán trên không gian 2D
- **Biểu đồ metrics**: Loss, Gradient Norm, Update Norm theo thời gian
- **Biểu đồ so sánh**: So sánh trực tiếp giữa các thuật toán

Tất cả biểu đồ được lưu trong thư mục `plots/`.

## 🔬 Chi tiết Kỹ thuật

### Hàm Kiểm tra

#### 1. Rosenbrock
$$f(x,y) = (a - x)^2 + b(y - x^2)^2$$

- Tham số mặc định: a=1, b=100
- Điểm cực tiểu: (1, 1)
- Đặc điểm: Thung lũng hẹp, khó tối ưu

#### 2. Ill-Conditioned Quadratic
$$f(x,y) = 0.5 \times (\kappa x^2 + y^2)$$

- Tham số mặc định: κ=100
- Điểm cực tiểu: (0, 0)
- Đặc điểm: Điều kiện xấu, hình elip dài

#### 3. Saddle Point
$$f(x,y) = 0.5 \times (x^2 - y^2)$$

- Điểm yên ngựa: (0, 0)
- Đặc điểm: Không có cực tiểu toàn cục

### Thuật toán Tối ưu

#### 1. SGD (Stochastic Gradient Descent)
```
θ_new = θ_old - lr × gradient
```

#### 2. SGD Momentum
```
v_new = β × v_old + gradient
θ_new = θ_old - lr × v_new
```

#### 3. RMSProp
```
s_new = ρ × s_old + (1-ρ) × gradient²
θ_new = θ_old - lr × gradient / √(s_new + ε)
```

#### 4. Adam
```
m_new = β₁ × m_old + (1-β₁) × gradient
v_new = β₂ × v_old + (1-β₂) × gradient²
m_hat = m_new / (1 - β₁^t)
v_hat = v_new / (1 - β₂^t)
θ_new = θ_old - lr × m_hat / (√v_hat + ε)
```

## 📈 Phân tích Kết quả

Mỗi thí nghiệm lưu trữ:
- **iteration**: Số vòng lặp
- **x, y**: Tọa độ tham số tại mỗi bước
- **loss**: Giá trị hàm mục tiêu
- **grad_norm**: Chuẩn của gradient
- **update_norm**: Chuẩn của bước cập nhật
- **grad_x, grad_y**: Các thành phần gradient

## 🎯 Tùy chỉnh

### Thay đổi cấu hình thí nghiệm

Chỉnh sửa hàm `create_experiment_configs()` trong `run_experiment.py`:

```python
# Thêm learning rate mới
optimizers = [
    {'type': 'Adam', 'params': {'lr': 0.0001}},  # Learning rate nhỏ hơn
    # ...
]

# Thay đổi số vòng lặp
num_iterations = 2000  # Tăng số vòng lặp
```

### Thêm hàm kiểm tra mới

Tạo lớp mới trong `test_functions.py`:

```python
class MyFunction(TestFunction):
    def compute(self, x, y):
        # Triển khai hàm của bạn
        return ...
    
    def gradient(self, x, y):
        # Triển khai gradient
        return grad_x, grad_y
    
    def hessian(self, x, y):
        # Triển khai Hessian
        return np.array([[h_xx, h_xy], [h_xy, h_yy]])
```

### Thêm thuật toán tối ưu mới

Tạo lớp mới trong `optimizers.py`:

```python
class MyOptimizer(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr
        # Khởi tạo trạng thái
    
    def step(self, params, gradients):
        # Triển khai logic cập nhật
        return new_x, new_y
    
    def reset(self):
        # Reset trạng thái
        pass
```

## 📝 Ví dụ Sử dụng Module

### Sử dụng trực tiếp trong code

```python
from test_functions import Rosenbrock
from optimizers import Adam

# Khởi tạo
func = Rosenbrock(a=1, b=100)
opt = Adam(lr=0.001)

# Điểm bắt đầu
x, y = -1.0, 2.0

# Tối ưu hóa
for i in range(1000):
    loss = func.compute(x, y)
    grad_x, grad_y = func.gradient(x, y)
    x, y = opt.step((x, y), (grad_x, grad_y))
    
    if i % 100 == 0:
        print(f"Iteration {i}: loss = {loss:.6f}")
```

## 🐛 Troubleshooting

### Lỗi: Module not found
```bash
# Đảm bảo bạn đang ở đúng thư mục
cd /workspaces/GDSearch

# Cài đặt lại dependencies
pip install -r requirements.txt
```

### Thư mục results trống
```bash
# Chạy thí nghiệm trước
python run_experiment.py
```

### Không hiển thị biểu đồ
```bash
# Kiểm tra matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

## 📚 Tài liệu Tham khảo

- **SGD**: Robbins & Monro (1951)
- **Momentum**: Polyak (1964)
- **RMSProp**: Tieleman & Hinton (2012)
- **Adam**: Kingma & Ba (2014)

## 🤝 Đóng góp

Để thêm tính năng mới hoặc báo lỗi, vui lòng:
1. Fork dự án
2. Tạo branch mới
3. Commit thay đổi
4. Tạo Pull Request

## 📄 License

Dự án này được phát hành dưới giấy phép MIT.

## 👤 Tác giả

Dự án được tạo ra để nghiên cứu và so sánh các thuật toán tối ưu hóa trong machine learning.

---

**Lưu ý**: Dự án này được thiết kế cho mục đích học tập và nghiên cứu. Để sử dụng trong production, cân nhắc thêm các tính năng như validation, error handling, và logging chi tiết hơn.

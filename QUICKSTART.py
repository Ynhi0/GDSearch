"""
Hướng dẫn nhanh sử dụng GDSearch
"""

# ============================================================
# HƯỚNG DẪN NHANH
# ============================================================

print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║        GDSearch - So Sánh Thuật Toán Tối Ưu Hóa         ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝

📦 BƯỚC 1: CÀI ĐẶT CÁC THƯ VIỆN
================================
    pip install -r requirements.txt

🧪 BƯỚC 2: CHẠY DEMO (TÙY CHỌN)
================================
    python demo.py

🚀 BƯỚC 3: CHẠY THÍ NGHIỆM ĐẦY ĐỦ
================================
    python run_experiment.py
    
    ⏱️  Thời gian ước tính: 2-5 phút
    📊 Kết quả: 72 file CSV trong thư mục results/

📈 BƯỚC 4: TẠO BIỂU ĐỒ
================================
    python plot_results.py
    
    🎨 Kết quả: Các biểu đồ PNG trong thư mục plots/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 CẤU TRÚC DỰ ÁN
================================

GDSearch/
├── 📄 test_functions.py      # 3 hàm kiểm tra (Rosenbrock, ...)
├── 📄 optimizers.py           # 4 optimizer (SGD, Adam, ...)
├── 📄 run_experiment.py       # Script chạy thí nghiệm
├── 📄 plot_results.py         # Script vẽ biểu đồ
├── 📄 demo.py                 # Demo nhanh
├── 📄 requirements.txt        # Thư viện phụ thuộc
├── 📁 results/                # Kết quả CSV
└── 📁 plots/                  # Biểu đồ PNG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 CÁC HÀM KIỂM TRA
================================
1️⃣  Rosenbrock          - Thung lũng hẹp
2️⃣  IllConditionedQuad  - Điều kiện xấu  
3️⃣  SaddlePoint         - Điểm yên ngựa

🤖 CÁC THUẬT TOÁN TỐI ƯU
================================
1️⃣  SGD                 - Gradient Descent cơ bản
2️⃣  SGDMomentum         - SGD với Momentum
3️⃣  RMSProp             - Adaptive learning rate
4️⃣  Adam                - Adaptive Moments

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 SỬ DỤNG NÂNG CAO
================================

# Import các module
from test_functions import Rosenbrock
from optimizers import Adam

# Khởi tạo
func = Rosenbrock(a=1, b=100)
opt = Adam(lr=0.001)

# Tối ưu hóa
x, y = -1.0, 2.0  # Điểm bắt đầu

for i in range(1000):
    loss = func.compute(x, y)
    grad_x, grad_y = func.gradient(x, y)
    x, y = opt.step((x, y), (grad_x, grad_y))
    
    if i % 100 == 0:
        print(f"Iter {i}: loss = {loss:.6f}")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 TROUBLESHOOTING
================================

❌ Lỗi: ModuleNotFoundError
   ➜ Chạy: pip install -r requirements.txt

❌ Thư mục results/ trống
   ➜ Chạy: python run_experiment.py

❌ Không có biểu đồ
   ➜ Chạy: python plot_results.py sau khi có results/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 TÀI LIỆU THAM KHẢO
================================
- README_PROJECT.md     # Tài liệu chi tiết
- demo.py              # Ví dụ sử dụng
- Docstrings in code   # Mô tả hàm/class

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ TÍNH NĂNG NỔI BẬT
================================
✅ Module hóa tốt - Dễ mở rộng
✅ Tài liệu đầy đủ - Docstrings chi tiết
✅ Type hints - Code rõ ràng
✅ Tái tạo được - Random seed control
✅ Trực quan hóa - Biểu đồ đẹp mắt
✅ So sánh đa chiều - 72 thí nghiệm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 HỌC THÊM
================================
Để hiểu sâu hơn về các thuật toán:
- SGD: Robbins & Monro (1951)
- Momentum: Polyak (1964)  
- RMSProp: Tieleman & Hinton (2012)
- Adam: Kingma & Ba (2014)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🙏 CHÚC BẠN THÀNH CÔNG!
================================
""")

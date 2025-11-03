# Visualization Tools

Công cụ tạo biểu đồ chất lượng cao từ kết quả thí nghiệm.

## 📊 Available Scripts

### 1. `create_separate_plots.py`

Tạo 6 biểu đồ riêng biệt từ kết quả thí nghiệm multi-seed, mỗi biểu đồ được lưu dưới dạng file PNG độ phân giải cao (300 DPI).

#### Cách sử dụng:

**Cơ bản:**
```bash
python src/visualization/create_separate_plots.py
```

**Với custom paths:**
```bash
python src/visualization/create_separate_plots.py \
    --summary results/optimizer_summary.csv \
    --stats results/statistical_comparisons.csv \
    --detailed results/multiseed_detailed.csv \
    --output plots
```

#### Output:

Script sẽ tạo 6 file PNG trong thư mục `plots/`:

1. **`01_final_loss_comparison.png`** (140KB)
   - So sánh final loss giữa các optimizer
   - Bar chart với error bars (mean ± std)
   - Log scale cho trục Y
   - **Mục đích:** Thấy rõ optimizer nào đạt loss thấp nhất

2. **`02_distance_to_optimum.png`** (126KB)
   - Khoảng cách từ điểm cuối đến optimum (1,1)
   - Bar chart với error bars
   - **Mục đích:** Đánh giá độ chính xác của convergence

3. **`03_convergence_rate.png`** (116KB)
   - Tỷ lệ convergence thành công trên 5 seeds
   - Bar chart với phần trăm
   - **Mục đích:** Độ tin cậy của optimizer (reliability)

4. **`04_loss_distribution_boxplot.png`** (119KB)
   - Box plot phân phối loss qua các seeds
   - Hiển thị median (red), mean (blue), quartiles
   - Log scale
   - **Mục đích:** Xem variance và outliers

5. **`05_statistical_significance_heatmap.png`** (213KB)
   - Ma trận p-values cho tất cả các cặp optimizer
   - Green = significant (p<0.05), Red = not significant
   - Annotations: *** (p<0.001), ** (p<0.01), * (p<0.05), ns
   - **Mục đích:** Xem các cặp nào khác biệt có ý nghĩa thống kê

6. **`06_effect_sizes.png`** (183KB)
   - Cohen's d effect sizes cho các so sánh
   - Color-coded: small, medium, large, very large
   - Reference lines tại ±0.2, ±0.5, ±0.8
   - **Mục đích:** Đánh giá độ lớn của khác biệt (practical significance)

## 📝 Input Files Required

Script cần 3 CSV files từ thí nghiệm multi-seed:

### 1. `optimizer_summary.csv`
```csv
Optimizer,Mean Loss,Std Loss,Mean Distance,Converged
SGD,0.0221,0.0141,0.3009,0/5
SGD+Momentum,1.32e-08,1.76e-09,0.00026,4/5
...
```

### 2. `statistical_comparisons.csv`
```csv
Comparison,t-stat,p-value,Cohens d,Significant,Effect
SGD+Momentum vs SGD,-3.51,0.0080,-2.22,Yes,large
...
```

### 3. `multiseed_detailed.csv`
```csv
seed,final_loss,distance_to_optimum,iterations,converged,optimizer
42,0.00123,0.0795,2000,False,SGD
...
```

## 🎨 Design Principles

### Color Scheme:
- **SGD**: `#FF6B6B` (Coral red)
- **SGD+Momentum**: `#4ECDC4` (Turquoise)
- **RMSProp**: `#45B7D1` (Sky blue)
- **Adam**: `#FFA07A` (Light salmon)

### Typography:
- Title: 14pt, bold
- Axis labels: 12pt, bold
- Value labels: 9-11pt, bold
- DPI: 300 (publication quality)

### Guidelines:
- ✅ **High contrast** - easy to distinguish
- ✅ **Consistent colors** - same optimizer = same color
- ✅ **Clear labels** - no ambiguity
- ✅ **Error bars** - show uncertainty
- ✅ **Grid lines** - easy to read values

## 🔧 Customization

### Thay đổi màu sắc:

```python
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
# Đổi thành màu khác nếu muốn
```

### Thay đổi kích thước:

```python
plt.figure(figsize=(10, 6))  # width, height in inches
```

### Thay đổi DPI:

```python
plt.savefig(output_file, bbox_inches='tight', dpi=300)
# Tăng lên 600 nếu cần siêu nét, giảm xuống 150 nếu muốn file nhỏ
```

## 📊 Use Cases

### 1. Báo cáo NCKH:
- Dùng tất cả 6 plots trong phần Results
- Plot 1,2,3: Hiệu suất các optimizer
- Plot 4: Phân phối và variance
- Plot 5,6: Statistical evidence

### 2. Presentation slides:
- Plot 1: Overview slide - Final loss comparison
- Plot 3: Reliability slide - Convergence rate
- Plot 5: Statistics slide - Significance matrix

### 3. Paper submission:
- All plots are 300 DPI - đủ cho journal requirements
- Caption suggestions included in docstrings

### 4. Defense Q&A:
- Plot 4: Trả lời câu hỏi về variance
- Plot 6: Trả lời câu hỏi về practical significance
- Plot 5: Trả lời câu hỏi về statistical rigor

## 🚀 Quick Start

```bash
# 1. Run multi-seed experiment
python src/experiments/run_full_analysis.py --seeds 42,123,456,789,1024

# 2. Create separate plots
python src/visualization/create_separate_plots.py

# 3. View results
ls -lh plots/0*.png
```

## 💡 Tips

1. **Emoji warnings:** Nếu thấy warning về emoji fonts, ignore - không ảnh hưởng kết quả
2. **Log scale:** Plots 1 và 4 dùng log scale vì losses khác biệt nhiều bậc
3. **P-value heatmap:** Chỉ hiển thị upper/lower triangle (không duplicate)
4. **Effect sizes:** Negative = first optimizer better, Positive = second optimizer better

## 📚 References

- Cohen's d interpretation: small (0.2), medium (0.5), large (0.8)
- P-value significance: * (p<0.05), ** (p<0.01), *** (p<0.001)
- Box plot: Red line = median, Blue dashed = mean

## 🐛 Troubleshooting

**Problem:** `KeyError: 'optimizer'`
- **Solution:** Check CSV column names match exactly

**Problem:** Empty plots
- **Solution:** Verify CSV files exist and have data

**Problem:** Low resolution
- **Solution:** Increase DPI in script (default 300)

**Problem:** Fonts look weird
- **Solution:** Install DejaVu Sans font or change matplotlib font

---

**Author:** GDSearch Team  
**Last Updated:** November 3, 2025  
**Version:** 1.0

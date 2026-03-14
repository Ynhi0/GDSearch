# Chương 3. Phương pháp nghiên cứu

## 3.1. Định hướng phương pháp luận 

Phương pháp nghiên cứu trong báo cáo được xây dựng theo định hướng thực nghiệm có kiểm soát. Mục tiêu không dừng ở việc so sánh chỉ số đầu ra, mà tập trung làm rõ cơ chế động học của quá trình tối ưu. Câu hỏi trọng tâm là: những thành phần nào trong quy tắc cập nhật của thuật toán tối ưu bậc nhất chi phối đồng thời tốc độ hội tụ, độ ổn định quỹ đạo và năng lực tổng quát hóa, khi chuyển từ không gian tối ưu hình học đơn giản sang các bài toán học sâu thực tế. Để trả lời câu hỏi này, nghiên cứu triển khai một chuỗi bằng chứng liên thông qua ba tầng bối cảnh: phân tích trên hàm kiểm tra hai chiều, xác thực trên bộ dữ liệu phân loại chuẩn, và kiểm định trên tác vụ ứng dụng có độ phức tạp cao hơn. Trục xuyên suốt của chương là nguyên tắc thay đổi tối thiểu theo cơ chế, nhằm bảo đảm khả năng quy kết nguyên nhân và giảm nhiễu đồng biến do cấu hình.

### 3.1.1. Giả thuyết nghiên cứu và tiêu chí bác bỏ

Để mạch suy luận có thể kiểm định và có thể bác bỏ, chương này xác lập ba giả thuyết nghiên cứu tường minh. Giả thuyết thứ nhất $H_1$ cho rằng các cơ chế thích nghi theo moment bậc một và bậc hai đạt tốc độ hội tụ cao hơn SGD chuẩn khi ngân sách huấn luyện và cấu hình dữ liệu được giữ cố định [1]–[5]. Giả thuyết thứ hai $H_2$ cho rằng lợi thế hội tụ chỉ có ý nghĩa phương pháp luận khi đồng thời duy trì hoặc cải thiện chỉ số dự báo trên tập kiểm tra [8], [14], [15]. Giả thuyết thứ ba $H_3$ cho rằng hiệu ứng quan sát được có tính ổn định khi thay đổi hạt giống khởi tạo và khi dịch chuyển qua ba tầng bối cảnh thực nghiệm. Một giả thuyết chỉ được xem là được ủng hộ khi đồng thời thỏa ba điều kiện: có ý nghĩa thống kê sau điều chỉnh đa giả thuyết, có kích thước hiệu ứng đạt ngưỡng thực hành, và có tính nhất quán theo tổng hợp đa hạt giống.

## 3.2. Thiết lập toán học thống nhất

Cho tập dữ liệu huấn luyện $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^N$, mô hình tham số hóa $h_\theta$, và hàm mất mát mẫu $\ell(\cdot)$, bài toán tối ưu thực nghiệm được viết dưới dạng (3.1)

$$
\min_{\theta\in\mathbb{R}^d} f(\theta), \qquad f(\theta)=\frac{1}{N}\sum_{i=1}^N \ell\big(h_\theta(x_i),y_i\big).\tag{3.1}
$$

Tại bước lặp $t$, với minibatch $\mathcal{B}_t$, gradient ngẫu nhiên được mô hình hóa bởi (3.2)

$$
g_t = \nabla f_{\mathcal{B}_t}(\theta_t) = \nabla f(\theta_t) + \xi_t, \qquad
\mathbb{E}[\xi_t]=0, \qquad \mathrm{Var}(\xi_t)\propto \frac{1}{|\mathcal{B}_t|}.\tag{3.2}
$$

Quy tắc cập nhật SGD chuẩn là (3.3)

$$
θ_{t+1}=θ_t-η_t g_t.\tag{3.3}
$$

Với dạng Momentum theo Polyak, biến vận tốc cập nhật theo $v_{t+1}=\beta v_t+g_t$ và tham số cập nhật theo $\theta_{t+1}=\theta_t-\eta_t v_{t+1}$ [2]. Với RMSProp, moment bậc hai mũ được viết $s_{t+1}=\rho s_t+(1-\rho)(g_t\odot g_t)$, sau đó cập nhật $\theta_{t+1}=\theta_t-\eta_t g_t/(\sqrt{s_{t+1}}+\varepsilon)$ [3]. Với Adam, hai moment mũ được định nghĩa bởi $m_{t+1}=\beta_1m_t+(1-\beta_1)g_t$ và $v_{t+1}=\beta_2v_t+(1-\beta_2)(g_t\odot g_t)$, hiệu chỉnh lệch khởi tạo bằng $\hat m_{t+1}=m_{t+1}/(1-\beta_1^{t+1})$ và $\hat v_{t+1}=v_{t+1}/(1-\beta_2^{t+1})$, rồi cập nhật theo $\theta_{t+1}=\theta_t-\eta_t\hat m_{t+1}/(\sqrt{\hat v_{t+1}}+\varepsilon)$ [4]. AdamW dùng suy giảm trọng số tách rời qua $\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_t\hat m_{t+1}/(\sqrt{\hat v_{t+1}}+\varepsilon)$ [5].

## 3.3. Thiết kế thực nghiệm ba giai đoạn

Giai đoạn thứ nhất tập trung vào động học tối ưu trên hàm kiểm tra hai chiều, nơi địa hình mục tiêu có thể được quan sát trực tiếp để nhận diện các hiện tượng dao động, kẹt thung lũng hẹp, và ứng xử tại điểm yên. Các hàm được sử dụng bao gồm Rosenbrock với biểu thức $f(x,y)=(a-x)^2+b(y-x^2)^2$ [6], hàm bậc hai điều kiện xấu $f(x,y)=\tfrac{1}{2}(\kappa x^2+y^2)$ với $\kappa\gg1$, hàm yên ngựa $f(x,y)=\tfrac{1}{2}(x^2-y^2)$, và hàm Ackley hai chiều $f(x,y)=-a\exp\big(-b\sqrt{(x^2+y^2)/2}\big)-\exp\big((\cos(cx)+\cos(cy))/2\big)+a+e$ với tham số chuẩn $a=20$, $b=0.2$, $c=2\pi$ [7]. Mỗi run ghi đầy đủ trajectory, bao gồm giá trị mục tiêu, chuẩn gradient, chuẩn cập nhật, và thông tin độ cong cục bộ khi khả dụng.

Giai đoạn thứ hai chuyển sang bộ dữ liệu chuẩn trong học sâu để kiểm định khả năng chuyển giao kết luận từ không gian 2D sang bài toán dự báo. Nghiên cứu sử dụng MNIST và CIFAR-10 [8], [9], với kiến trúc cơ sở thuộc họ MLP/CNN và cấu hình mở rộng cho các thí nghiệm bổ sung. Hàm mất mát phân loại được thống nhất theo cross-entropy trong (3.4)

$$
\mathcal{L}_{\mathrm{CE}}=-\frac{1}{N}\sum_{i=1}^N\log p_\theta(y_i\mid x_i),
\qquad
p_\theta(y_i\mid x_i)=\frac{\exp(z_{y_i})}{\sum_{c=1}^{C}\exp(z_c)}.\tag{3.4}
$$

và chỉ số độ chính xác được định nghĩa trong (3.5)

$$
\mathrm{Acc}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat y_i=y_i).\tag{3.5}
$$

Cấu hình huấn luyện được chuẩn hóa bằng giao thức chia train, validation, test cố định, lưu checkpoint theo epoch, và ghi nhật ký đồng nhất để bảo toàn tính so sánh.

Giai đoạn thứ ba mở rộng sang các tình huống ứng dụng nhằm đánh giá độ bền vững của kết luận trong bối cảnh dữ liệu và kiến trúc phức tạp hơn. Với NLP, nghiên cứu dùng bài toán phân loại cảm xúc dựa trên bộ dữ liệu IMDB [10], mô hình Transformer [12] và các biến thể BERT [13] để phân tích động học tối ưu trong không gian tham số lớn. Với thị giác y tế, nghiên cứu dùng tác vụ phân vùng ảnh trên kiến trúc U-Net 2D [14], có thể triển khai trên bộ dữ liệu y sinh gọn nhẹ như MedMNIST v2 [11], trong đó chỉ số trung tâm là hệ số Dice trong (3.6) [15].

$$
\mathrm{Dice}=\frac{2\sum_i p_i g_i+\varepsilon}{\sum_i p_i+\sum_i g_i+\varepsilon}.\tag{3.6}
$$

Trong biểu thức này, $p_i$ là dự đoán nhị phân và $g_i$ là nhãn tham chiếu tại vị trí $i$. Dữ liệu thật được ưu tiên cho suy luận chính, trong khi dữ liệu tổng hợp chỉ dùng cho mục tiêu kiểm thử kỹ thuật quy trình thực nghiệm.

## 3.4. Quy tắc hội tụ, đo lường động học và chuẩn hóa đánh giá

Thay vì phụ thuộc vào một chỉ số đơn lẻ, nghiên cứu sử dụng tiêu chí hội tụ tổ hợp để tăng độ ổn định suy luận trước nhiễu số học. Cụ thể, tại thời điểm $t$, trạng thái hội tụ được xác định bởi điều kiện (3.7)

$$
\mathrm{Converged}(t;\tau)=\mathbf{1}\left(\|\nabla f_t\|_2<\tau\ \lor\ f_t<\tau\right).\tag{3.7}
$$

trong đó $\tau=10^{-3}$ là ngưỡng thực hành và $\tau=10^{-6}$ là ngưỡng nghiêm ngặt. Trong không gian 2D, độ điều kiện cục bộ được theo dõi thông qua phổ Hessian với chỉ số (3.8)

$$
\kappa_H=\frac{|\lambda_{\max}|}{\max(|\lambda_{\min}|,\varepsilon_H)}.\tag{3.8}
$$

Hệ thống đo lường bao gồm các chỉ số tối ưu như giá trị mất mát cuối kỳ (final loss), chuẩn gradient, chuẩn cập nhật, tốc độ hội tụ và tỉ lệ hội tụ theo ngưỡng; đồng thời bao gồm các chỉ số dự báo như Accuracy hoặc Dice tùy bài toán [8], [14], [15]. Cách tổ chức song song hai nhóm chỉ số cho phép đối chiếu trực tiếp giữa cơ chế động học và hiệu năng đầu ra.

### 3.4.1. Ánh xạ giả thuyết, chỉ số và phép kiểm định

Ánh xạ kiểm định được quy định trước để tránh diễn giải tùy ý sau quan sát. Với $H_1$, biến đáp ứng chính là số bước hoặc số epoch đạt điều kiện hội tụ; các chỉ số phụ gồm chuẩn gradient cuối và độ dao động quỹ đạo. So sánh cặp giữa các optimizer được thực hiện trên mẫu đa hạt giống trong cùng cấu hình. Với $H_2$, biến đáp ứng chính là chỉ số đầu ra trên tập kiểm tra, cụ thể là Accuracy cho phân loại và Dice cho phân vùng; giá trị mất mát cuối kỳ (final loss) được dùng như tín hiệu phụ để phát hiện đánh đổi không mong muốn giữa tối ưu hóa và tổng quát hóa. Với $H_3$, biến đáp ứng chính là độ ổn định liên hạt giống và liên bối cảnh, được lượng hóa qua độ phân tán và tính bền vững của hướng hiệu ứng giữa các tập dữ liệu và kiến trúc. Quy tắc kết luận thống nhất là chỉ chấp nhận một giả thuyết khi cả ba lớp bằng chứng cùng đồng thuận: ý nghĩa đã hiệu chỉnh, kích thước hiệu ứng và độ ổn định đa hạt giống.

## 3.5. Thiết kế cắt lớp cơ chế và công bằng cấu hình

Thiết kế cắt lớp của chương được tổ chức theo nguyên tắc “khác biệt tối thiểu theo cơ chế”, nghĩa là chỉ thay đổi thành phần cập nhật cần kiểm định trong khi khóa các thành phần còn lại để bảo toàn khả năng quy kết nguyên nhân. Chuỗi đối sánh được xây dựng theo lộ trình SGD $\rightarrow$ Momentum $\rightarrow$ RMSProp $\rightarrow$ Adam $\rightarrow$ AdamW, phản ánh tiến hóa lịch sử của họ thuật toán tối ưu bậc nhất [1]–[5]. Với mỗi cặp đối sánh, báo cáo giữ bất biến tập dữ liệu, chiến lược tiền xử lý, kiến trúc mô hình, lịch học, bộ dừng, và giao thức đánh giá để khác biệt quan sát được có thể quy về cơ chế tối ưu thay vì quy trình huấn luyện.

Ở lớp kiểm soát thực nghiệm, mọi lần chạy đều dùng cùng tập hạt giống cho toàn bộ optimizer trong cùng bài toán, cùng số epoch tối đa, cùng điều kiện dừng sớm, cùng chu kỳ checkpoint và cùng cấu trúc nhật ký cho loss, chuẩn gradient, chuẩn cập nhật và chỉ số dự báo. Cách khóa biến này tạo ra không gian so sánh đẳng điều kiện. Theo đó, một cấu hình chỉ được xem là “hợp lệ để suy luận” khi không vi phạm ràng buộc kỹ thuật như lỗi số học, thiếu nhật ký bắt buộc hoặc gián đoạn checkpoint. Phần dữ liệu không hợp lệ vẫn được giữ cho mục đích kiểm toán hệ thống, nhưng loại khỏi tập suy luận hiệu ứng để tránh làm sai lệch kết luận thống kê.

Ở lớp công bằng siêu tham số, chương áp dụng chính sách ngân sách đối xứng giữa các optimizer, thay vì cho phép một thuật toán được tìm kiếm rộng hơn thuật toán khác. Cụ thể, mỗi optimizer nhận cùng tổng số cấu hình thử và cùng tổng chi phí huấn luyện cho hai pha: quét thô và tinh chỉnh cục bộ. Cấu hình cuối được chọn theo trung vị hiệu năng đa hạt giống, sau đó mới xét độ phân tán như tiêu chí phụ để ưu tiên tính ổn định. Về triển khai, chiến lược tìm kiếm được chuẩn hóa qua framework tối ưu siêu tham số có kiểm soát ngân sách như Optuna [17]. Chính sách này giúp tránh tình huống một optimizer được lợi do “tuning quá mức”, đồng thời nhất quán với khung ra quyết định đa bằng chứng ở Thuật toán 3.6 và 3.7.

Để tăng độ chặt phương pháp luận, báo cáo dùng quy tắc chấp nhận cấu hình theo ba tầng. Tầng thứ nhất là tính khả thi kỹ thuật, yêu cầu lần chạy hoàn tất và dữ liệu nhật ký đầy đủ. Tầng thứ hai là tính hợp lệ thống kê, yêu cầu kết quả có thể đi vào quy trình hiệu chỉnh đa giả thuyết mà không vi phạm điều kiện kiểm định. Tầng thứ ba là ý nghĩa thực hành, yêu cầu kích thước hiệu ứng đạt ngưỡng tối thiểu do nghiên cứu đặt trước. Chỉ các cấu hình vượt qua đủ ba tầng mới được đưa vào phần tổng hợp kết luận giữa các optimizer.

## 3.6. Thuật toán và mã giả của quy trình thực nghiệm

Phần này trình bày các thuật toán ở mức vận hành để bảo đảm phương pháp không chỉ đúng về khái niệm mà còn khả thi khi triển khai. Mục tiêu của lớp thuật toán là chuyển thiết kế nghiên cứu thành quy trình có thể lặp lại, kiểm thử và kiểm toán. Trong toàn chương, các thuật ngữ kỹ thuật được thống nhất theo quy ước: run là lần chạy, seed là hạt giống, checkpoint là điểm lưu trạng thái, artifact là tệp kết quả trung gian hoặc cuối. Thuật toán 3.1 mô tả quy trình thực nghiệm tổng quát; Thuật toán 3.2 mô tả lớp tổng hợp và suy luận thống kê sau huấn luyện; Thuật toán 3.3 mô tả cơ chế tái lập và phục hồi lỗi ở mức hệ thống; Thuật toán 3.4 chuẩn hóa chính sách công bằng siêu tham số; Thuật toán 3.5 xác định quy tắc hội tụ và dừng huấn luyện; Thuật toán 3.6 mô tả luồng kiểm định có điều chỉnh đa giả thuyết; và Thuật toán 3.7 chuẩn hóa quy trình ra quyết định chấp nhận hoặc bác bỏ giả thuyết nghiên cứu.

### 3.6.1. Thuật toán 3.1: Quy trình thực nghiệm đa giai đoạn

Thuật toán 3.1 mô tả vòng đời đầy đủ của một đợt thực nghiệm, từ cấu hình ban đầu đến báo cáo cuối. Tại mỗi bước, hệ thống duy trì tính truy vết giữa cấu hình, seed và artifact đầu ra.

```text
Thuật toán 3.1: Quy trình thực nghiệm đa giai đoạn
Đầu vào: Tập bài toán P, tập optimizer O, tập seed S, ngân sách huấn luyện T
Đầu ra: Nhật ký chuẩn hóa L, báo cáo tổng hợp R

1: Khởi tạo môi trường xác định; tạo cấu trúc lưu log L và kho artifact
2: Với mỗi bài toán p thuộc P:
3:     Chuẩn hóa dữ liệu và tách train/validation/test theo giao thức cố định
4:     Với mỗi optimizer o thuộc O:
5:         h_o $\leftarrow$ Chọn siêu tham số theo chính sách công bằng
6:         Với mỗi seed s thuộc S:
7:             Thiết lập seed toàn cục bằng s
8:             Khởi tạo mô hình theta_0 và trạng thái optimizer state_0
9:             Với t từ 1 đến T:
10:                B_t $\leftarrow$ Lấy minibatch từ train set
11:                g_t $\leftarrow$ Tính gradient trên B_t
12:                (theta_t, state_t) $\leftarrow$ Cập nhật theo quy tắc của optimizer o
13:                Ghi vào L: loss, ||g_t||, ||Delta theta_t||, metric validation
14:                Nếu gặp lỗi số học hoặc lỗi tài nguyên thì:
15:                    Gắn nhãn run_status = failed/tainted; lưu artifact tạm; thoát vòng lặp thời gian
16:            Kết thúc vòng t
17:            Nếu run_status chưa bị lỗi thì đánh giá trên test set và gắn nhãn successful
18:            Lưu checkpoint cuối, metadata cấu hình, và artifact theo (p, o, s)
19:        Kết thúc vòng seed
20:        Tổng hợp kết quả đa seed cho cặp (p, o)
21:     Kết thúc vòng optimizer
22: Kết thúc vòng bài toán
23: Thực hiện so sánh liên-optimizer có điều chỉnh đa giả thuyết
24: Xuất R gồm bảng, hình và metadata tái lập
25: Trả về (L, R)
```

### 3.6.2. Thuật toán 3.2: Tổng hợp đa seed và so sánh liên-optimizer

Thuật toán 3.2 tập trung vào lớp suy luận sau huấn luyện, trong đó dữ liệu đa seed được chuẩn hóa về cùng không gian metric và được dùng cho suy luận hiệu ứng. Điểm quan trọng của thuật toán này là tách rõ thống kê mô tả, kiểm định giả thuyết và diễn giải kích thước hiệu ứng.

```text
Thuật toán 3.2: Tổng hợp đa seed và suy luận so sánh cặp
Đầu vào: Bảng kết quả theo run D, mức ý nghĩa alpha
Đầu ra: Ma trận so sánh cặp M, tập kết luận hiệu chỉnh C

1: D_valid $\leftarrow$ Lọc D theo run_status ∈ {successful}
2: T $\leftarrow$ Trích metric cuối kỳ cho từng bộ (problem, optimizer, seed)
3: Tính thống kê mô tả cho từng (problem, optimizer): median, mean, std, P10, P90
4: Khởi tạo danh sách phép thử Q rỗng
5: Với mỗi problem p:
6:     O_p $\leftarrow$ Tập optimizer có dữ liệu hợp lệ trên p
7:     Với mỗi cặp (o_i, o_j) thuộc O_p, i < j:
8:         x $\leftarrow$ Mẫu metric seed-level của (p, o_i)
9:         y $\leftarrow$ Mẫu metric seed-level của (p, o_j)
10:        (p_raw, test_name) $\leftarrow$ Chạy kiểm định phù hợp trên (x, y)
11:        d $\leftarrow$ Tính kích thước hiệu ứng (ví dụ Cohen's d)
12:        Thêm bản ghi (p, o_i, o_j, p_raw, d, test_name) vào Q
13:    Kết thúc vòng cặp
14: Kết thúc vòng problem
15: p_adj $\leftarrow$ Điều chỉnh đa giả thuyết cho toàn bộ p_raw trong Q
16: M $\leftarrow$ Dựng ma trận so sánh từ (p_adj, d, hướng hiệu ứng)
17: C $\leftarrow$ Sinh kết luận theo quy tắc (ý nghĩa đã hiệu chỉnh + kích thước hiệu ứng + độ ổn định)
18: Trả về (M, C)
```

### 3.6.3. Thuật toán 3.3: Quản lý tính tái lập và phục hồi lỗi

Thuật toán 3.3 mô tả cơ chế kiểm soát độ tin cậy kỹ thuật. Cơ chế này giúp tách lỗi hệ thống ra khỏi tín hiệu thuật toán, đồng thời giữ quy trình có thể tiếp tục sau sự cố tài nguyên.

```text
Thuật toán 3.3: Thực thi chịu lỗi và bảo toàn tính tái lập
Đầu vào: Ngữ cảnh run X
Đầu ra: Gói artifact đã xác thực A

1: env_ok $\leftarrow$ Kiểm tra dấu vân tay môi trường (phiên bản thư viện, thiết bị, cấu hình hệ)
2: Nếu env_ok = false thì trả về lỗi cấu hình môi trường
3: Lưu snapshot cấu hình và seed trước khi huấn luyện
4: status $\leftarrow$ running
5: Trong quá trình huấn luyện, lặp theo chu kỳ checkpoint:
6:     Lưu checkpoint mô hình, trạng thái optimizer, và log trung gian
7:     anomaly $\leftarrow$ Phát hiện NaN/Inf, OOM, gián đoạn I/O, hoặc vi phạm schema log
8:     Nếu anomaly = true thì:
9:         status $\leftarrow$ failed hoặc tainted tùy loại lỗi
10:        Ghi mã lỗi tường minh; lưu artifact từng phần và metadata phục hồi
11:        Thoát vòng huấn luyện
12: Kết thúc chu kỳ
13: Nếu status = running thì status $\leftarrow$ successful
14: A $\leftarrow$ Đóng gói artifact cuối cùng theo schema chuẩn
15: Kiểm tra tính tương thích schema của A cho bước tổng hợp hạ nguồn
16: Trả về A cùng trạng thái status
```

### 3.6.4. Thuật toán 3.4: Chính sách công bằng siêu tham số liên-optimizer
Để bảo đảm tính công bằng của so sánh, chương áp dụng một thủ tục lựa chọn siêu tham số nhất quán giữa các optimizer. Thủ tục này ưu tiên thiết kế ngân sách đối xứng, tách pha quét thô và pha tinh chỉnh, đồng thời khóa các điều kiện ngoài optimizer để tránh sai lệch cấu trúc.

```text
Thuật toán 3.4: Chính sách công bằng siêu tham số liên-optimizer
Đầu vào: Tập optimizer O, không gian tìm kiếm H, ngân sách thử nghiệm B, tập seed S_ref
Đầu ra: Bộ siêu tham số công bằng H*

1: Chia ngân sách B thành B_1 (quét thô) và B_2 (tinh chỉnh)
2: Khởi tạo H* rỗng
3: Với mỗi optimizer o thuộc O:
4:     H_o $\leftarrow$ Xây dựng không gian con tương đương về độ biểu đạt
5:     U_1 $\leftarrow$ Lấy tập ứng viên từ H_o theo ngân sách B_1
6:     Đánh giá từng ứng viên trong U_1 trên cùng tập seed S_ref
7:     R_o $\leftarrow$ Chọn vùng ứng viên hứa hẹn theo tiêu chí (hội tụ, ổn định, chi phí)
8:     U_2 $\leftarrow$ Tinh chỉnh cục bộ trong R_o theo ngân sách B_2
9:     h_o* $\leftarrow$ Chọn cấu hình tối ưu theo trung vị hiệu năng và độ bền vững đa seed
10:    H* $\leftarrow$ H* ∪ { (o, h_o*) }
11: Kết thúc vòng optimizer
12: Khóa H* và dùng cố định cho toàn bộ thí nghiệm chính
13: Trả về H*
```

### 3.6.5. Thuật toán 3.5: Quy tắc xác định hội tụ và dừng huấn luyện

Vì tiêu chí hội tụ phụ thuộc đặc điểm bài toán, chương sử dụng quy tắc tổ hợp giữa tín hiệu động học và tín hiệu mục tiêu. Cách này giúp tránh kết luận hội tụ giả khi một chỉ báo đơn lẻ dao động bất thường.

```text
Thuật toán 3.5: Quy tắc xác định hội tụ và dừng huấn luyện
Đầu vào: Chuỗi loss F_t, chuẩn gradient ||g_t||, ngưỡng (epsilon_F, epsilon_g), cửa sổ w, số cửa sổ liên tiếp k
Đầu ra: Nhãn trạng thái hội tụ z

1: stable_count $\leftarrow$ 0; z $\leftarrow$ đang_huấn_luyện
2: Với mỗi thời điểm t >= w:
3:     delta_F $\leftarrow$ median(|F_i - F_{i-1}|) trên cửa sổ [t-w+1, t]
4:     g_med $\leftarrow$ median(||g_i||) trên cửa sổ [t-w+1, t]
5:     cond_F $\leftarrow$ (delta_F <= epsilon_F)
6:     cond_g $\leftarrow$ (g_med <= epsilon_g)
7:     Nếu cond_F và cond_g thì stable_count $\leftarrow$ stable_count + 1
8:     Ngược lại stable_count $\leftarrow$ 0
9:     Nếu stable_count >= k thì:
10:        z $\leftarrow$ hội_tụ_ổn_định
11:        Dừng huấn luyện sớm
12: Kết thúc vòng thời gian
13: Nếu z vẫn là đang_huấn_luyện khi hết ngân sách thì z $\leftarrow$ chưa_hội_tụ_trong_ngân_sách
14: Trả về z
```

### 3.6.6. Thuật toán 3.6: Quy trình suy luận thống kê có điều chỉnh đa giả thuyết

Để bảo đảm tính chặt chẽ suy luận, chương chuẩn hóa luồng kiểm định từ tiền xử lý mẫu, ước lượng hiệu ứng, điều chỉnh đa giả thuyết đến kết luận cuối. Quy trình này nhấn mạnh diễn giải theo mức độ bằng chứng thay cho quyết định nhị phân.

```text
Thuật toán 3.6: Suy luận thống kê có điều chỉnh đa giả thuyết
Đầu vào: Bảng mẫu theo seed D, mức ý nghĩa alpha, tập phép kiểm định T
Đầu ra: Tập kết luận thống kê đã hiệu chỉnh C_hat

1: D_clean $\leftarrow$ Loại mẫu lỗi kỹ thuật và mẫu không hợp lệ
2: Khởi tạo danh sách kết quả S rỗng và danh sách p-value P rỗng
3: Với mỗi bài toán p trong D_clean:
4:     Với mỗi cặp optimizer (o_i, o_j):
5:         x $\leftarrow$ Mẫu seed-level của (p, o_i)
6:         y $\leftarrow$ Mẫu seed-level của (p, o_j)
7:         Nếu mẫu ghép cặp thì:
8:             Nếu sai khác gần chuẩn thì test $\leftarrow$ kiểm định t ghép cặp
9:             Ngược lại test $\leftarrow$ Wilcoxon signed-rank
10:        Ngược lại:
11:            Nếu mẫu gần chuẩn và phương sai không đồng nhất thì test $\leftarrow$ kiểm định t Welch
12:            Nếu mẫu gần chuẩn và phương sai đồng nhất thì test $\leftarrow$ kiểm định t hai mẫu độc lập
13:            Ngược lại test $\leftarrow$ Mann-Whitney U
14:        (p_raw, stat) $\leftarrow$ Thực hiện kiểm định test trên (x, y)
15:        eff $\leftarrow$ Tính kích thước hiệu ứng
16:        Lưu bản ghi r = (p, o_i, o_j, p_raw, stat, eff) vào S
17:        Thêm p_raw vào P
18:    Kết thúc vòng cặp
19: Kết thúc vòng bài toán
20: p_adj $\leftarrow$ Điều chỉnh đa giả thuyết cho P bằng thủ tục đã chọn (Holm/FDR)
21: Gán p_adj vào từng bản ghi trong S
22: Với mỗi bản ghi trong S:
23:     sig $\leftarrow$ (p_adj <= alpha)
24:     strength $\leftarrow$ Phân loại mức bằng chứng từ (sig, eff, độ ổn định đa seed)
25:     Thêm vào C_hat
26: Trả về C_hat
```

### 3.6.7. Thuật toán 3.7: Quy trình ra quyết định giả thuyết nghiên cứu

Để khép kín chuỗi suy luận từ dữ liệu đến kết luận khoa học, cần một lớp thuật toán riêng cho quyết định chấp nhận hoặc bác bỏ giả thuyết. Thuật toán này tổng hợp đầu ra từ lớp thống kê, lớp kích thước hiệu ứng và lớp ổn định đa hạt giống để tạo quyết định nhất quán giữa các bài toán.

```text
Thuật toán 3.7: Quy trình ra quyết định giả thuyết nghiên cứu
Đầu vào: Kết quả hiệu chỉnh C_hat, ngưỡng kích thước hiệu ứng delta_min, ngưỡng ổn định s_min
Đầu ra: Quyết định cho từng giả thuyết trong {H_1, H_2, H_3}

1: Với mỗi giả thuyết H_k:
2:     Lấy tập bằng chứng E_k liên quan đến H_k từ C_hat
3:     sig_ok $\leftarrow$ Tỉ lệ kết quả có ý nghĩa đã hiệu chỉnh trong E_k đạt ngưỡng yêu cầu
4:     eff_ok $\leftarrow$ Tỉ lệ kết quả có |effect_size| >= delta_min đạt ngưỡng yêu cầu
5:     stab_ok $\leftarrow$ Chỉ số ổn định đa hạt giống và đa bối cảnh >= s_min
6:     Nếu sig_ok và eff_ok và stab_ok thì
7:         decision(H_k) $\leftarrow$ Ủng_hộ
8:     Ngược lại nếu chỉ thỏa một phần điều kiện thì
9:         decision(H_k) $\leftarrow$ Bằng_chứng_chưa_đủ
10:    Ngược lại
11:        decision(H_k) $\leftarrow$ Bác_bỏ
12: Kết thúc vòng giả thuyết
13: Trả về decision(H_1), decision(H_2), decision(H_3)
```

## 3.7. Suy luận thống kê và kiểm soát độ tin cậy

Khung suy luận thống kê của chương ưu tiên ước lượng hiệu ứng và bất định, thay vì diễn giải nhị phân theo một ngưỡng p-value duy nhất. Ở cấp hạt giống, kết quả được mô tả bằng các thước đo trung tâm bền vững và độ phân tán. Ở cấp đối sánh cặp, mỗi phép kiểm định đi kèm điều chỉnh đa giả thuyết để kiểm soát tích lũy sai lầm loại I [23], [24]. Ở cấp kết luận, diễn giải luôn gắn với kích thước hiệu ứng và khoảng biến thiên quan sát được, nhằm phản ánh ý nghĩa thực hành. Quy tắc chọn phép kiểm định được khóa trước khi phân tích: kiểm tra tính chuẩn bằng Shapiro–Wilk trên sai khác ghép cặp hoặc trên từng mẫu khi không ghép cặp [18], kiểm tra đồng nhất phương sai bằng Levene trong trường hợp không ghép cặp [19], rồi chọn phép kiểm định theo luồng đã nêu ở Thuật toán 3.6 (Wilcoxon, Mann–Whitney U, Welch t-test) [20]–[22] để loại trừ nguy cơ chọn kiểm định theo kết quả mong muốn. Cách tổ chức này giúp hạn chế hiện tượng “p-hacking” và nâng độ tin cậy của phát hiện thực nghiệm [25].

## 3.8. Các đe dọa đến tính hợp lệ và biện pháp giảm thiểu

Phần này phân tách rõ các lớp đe dọa tính hợp lệ để bảo đảm kết luận của chương có thể đứng vững trước phản biện phương pháp luận.

### 3.8.1. Tính hợp lệ nội tại

Các mối đe dọa đối với tính hợp lệ nội tại chủ yếu đến từ sai khác cấu hình huấn luyện giữa các optimizer, nhiễu do hạt giống khởi tạo và lỗi số học trong quá trình tối ưu. Biện pháp giảm thiểu gồm chuẩn hóa giao thức huấn luyện, khóa ngân sách tính toán theo từng nhánh so sánh, duy trì chính sách công bằng siêu tham số và loại trừ các lần chạy không hợp lệ theo cơ chế trạng thái lần chạy. Ngoài ra, checkpoint định kỳ giúp giảm nguy cơ mất dữ liệu thí nghiệm và hạn chế sai lệch do dừng chạy ngoài ý muốn.

### 3.8.2. Tính hợp lệ cấu trúc

Các mối đe dọa đối với tính hợp lệ cấu trúc phát sinh khi khái niệm “tối ưu tốt” bị quy giản vào một chỉ số duy nhất. Nghiên cứu giảm thiểu rủi ro này bằng cách sử dụng song song chỉ số động học và chỉ số dự báo: lớp động học phản ánh hành vi thuật toán, còn lớp dự báo phản ánh chất lượng mô hình. Tiêu chí hội tụ tổ hợp dựa trên chuẩn gradient hoặc giá trị hàm mục tiêu cũng được áp dụng để tránh diễn giải lệch do đặc trưng của một chỉ báo đơn lẻ.

### 3.8.3. Tính hợp lệ ngoại tại

Các mối đe dọa đối với tính hợp lệ ngoại tại xuất hiện khi kết luận chỉ dựa trên một miền dữ liệu hoặc một họ mô hình. Thiết kế đa giai đoạn của chương xử lý điểm này bằng cách kiểm chứng trên các lớp bài toán có độ phức tạp tăng dần, từ hàm kiểm tra 2D đến bộ dữ liệu học sâu chuẩn và cuối cùng là tình huống ứng dụng. Cách tiếp cận này không loại bỏ hoàn toàn giới hạn khái quát hóa, nhưng tạo ra bằng chứng mạnh hơn cho tính ổn định của các xu hướng quan sát.

### 3.8.4. Tính hợp lệ kết luận thống kê

Các mối đe dọa đối với tính hợp lệ kết luận thống kê thường đến từ cỡ mẫu hạt giống nhỏ, số lượng so sánh cặp lớn và xu hướng diễn giải nhị phân theo p-value. Nghiên cứu giảm thiểu bằng tổng hợp đa hạt giống, báo cáo bổ sung kích thước hiệu ứng và độ phân tán, đồng thời điều chỉnh đa giả thuyết trong so sánh liên-optimizer. Kết luận cuối chỉ được chấp nhận khi tín hiệu thống kê đồng thuận với tín hiệu động học và tín hiệu hiệu năng, thay vì dựa trên một phép kiểm định đơn lẻ.

## 3.9. Chuẩn báo cáo và nguyên tắc tái lập

Mọi kết quả trong chương được trình bày theo chuẩn truy vết từ lần chạy thô đến tổng hợp cuối. Dữ liệu huấn luyện, cấu hình optimizer, hạt giống, trạng thái lần chạy và tệp kết quả trung gian đều được lưu theo cấu trúc nhất quán để có thể kiểm chứng chéo. Về trình bày, chương ưu tiên lập luận dưới dạng đoạn văn liên tục; trong đó công thức, giả định và quy tắc suy luận được nêu tường minh để người đọc có thể tái tạo logic phương pháp mà không phụ thuộc vào mô tả thủ tục rời rạc. Cấu trúc này hướng tới chuẩn của một báo cáo khoa học hoàn chỉnh, trong đó tính rõ ràng về giả định và tính tái lập là điều kiện tiên quyết cho giá trị khoa học của kết luận.

## 3.10. Cấu hình triển khai và kiểm soát sai lệch thực nghiệm

Để tăng mức độ hoàn chỉnh của phương pháp, nghiên cứu định nghĩa rõ lớp cấu hình triển khai gồm môi trường phần mềm, chính sách seed, cơ chế checkpoint và chiến lược xử lý lỗi. Ở lớp phần mềm, toàn bộ quy trình được chạy trên cùng một stack thư viện để tránh sai khác do phiên bản phụ thuộc [16]; ở lớp ngẫu nhiên, seed được truyền xuyên suốt từ bước khởi tạo dữ liệu đến bước huấn luyện nhằm giảm sai lệch giữa các lần chạy; ở lớp huấn luyện, trạng thái mô hình và các chỉ số trung gian được checkpoint theo epoch để bảo toàn khả năng phục hồi; và ở lớp độ tin cậy, run phát sinh lỗi số học hoặc ngắt tài nguyên được gắn nhãn trạng thái riêng để không làm ô nhiễm tập kết quả hợp lệ. Cách tổ chức này nhằm tách bạch rõ ràng giữa lỗi hệ thống và khác biệt thuật toán, qua đó giữ cho so sánh phương pháp có ý nghĩa khoa học.

## 3.11. Quy tắc tổng hợp đa seed và diễn giải kết luận

Ở cấp độ phân tích, báo cáo áp dụng quy tắc tổng hợp đa seed để giảm độ nhạy với khởi tạo ngẫu nhiên. Với mỗi cấu hình, kết quả được tổng hợp theo trung vị, trung bình và độ phân tán, đồng thời báo cáo thêm các phân vị để mô tả hình dạng phân phối hiệu năng. Khi thực hiện so sánh cặp giữa các optimizer, kết luận không dựa đơn thuần vào p-value thô mà đi kèm điều chỉnh đa giả thuyết và diễn giải theo kích thước hiệu ứng. Quy tắc diễn giải cuối cùng yêu cầu sự nhất quán giữa ba lớp bằng chứng, gồm động học tối ưu, metric đầu ra và độ ổn định đa seed; chỉ khi ba lớp này đồng thuận thì kết luận mới được xem là đủ mạnh để chuyển sang chương kết quả và thảo luận.

## Tài liệu tham khảo (IEEE)

[1] H. Robbins and S. Monro, “A stochastic approximation method,” *The Annals of Mathematical Statistics*, vol. 22, no. 3, pp. 400–407, 1951. doi: 10.1214/aoms/1177729586.

[2] B. T. Polyak, “Some methods of speeding up the convergence of iteration methods,” *USSR Computational Mathematics and Mathematical Physics*, vol. 4, no. 5, pp. 1–17, 1964. doi: 10.1016/0041-5553(64)90137-5.

[3] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.

[4] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. 3rd Int. Conf. Learn. Representations (ICLR)*, San Diego, CA, USA, 2015. [Online]. Available: https://openreview.net/forum?id=ryQu7f-RZ

[5] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[6] H. H. Rosenbrock, “An automatic method for finding the greatest or least value of a function,” *The Computer Journal*, vol. 3, no. 3, pp. 175–184, 1960. doi: 10.1093/comjnl/3.3.175.

[7] D. H. Ackley, *A Connectionist Machine Approach to Genetic Hillclimbing*. Boston, MA, USA: Springer, 1987.

[8] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998. doi: 10.1109/5.726791.

[9] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778. doi: 10.1109/CVPR.2016.90.

[10] A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts, “Learning word vectors for sentiment analysis,” in *Proc. 49th Annu. Meeting ACL: Human Language Technologies*, Portland, OR, USA, 2011, pp. 142–150. [Online]. Available: https://aclanthology.org/P11-1015/

[11] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni, “MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification,” *Scientific Data*, vol. 10, no. 1, p. 41, 2023. doi: 10.1038/s41597-022-01721-8.

[12] A. Vaswani *et al*., “Attention is all you need,” in *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, 2017. [Online]. Available: https://papers.nips.cc/paper/7181-attention-is-all-you-need

[13] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in *Proc. 2019 Conf. North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, Minneapolis, MN, USA, 2019, pp. 4171–4186. doi: 10.18653/v1/N19-1423.

[14] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in *Proc. 18th Int. Conf. Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Munich, Germany, 2015, pp. 234–241. doi: 10.1007/978-3-319-24574-4_28.

[15] L. R. Dice, “Measures of the amount of ecologic association between species,” *Ecology*, vol. 26, no. 3, pp. 297–302, 1945. doi: 10.2307/1932409.

[16] A. Paszke *et al*., “PyTorch: An imperative style, high-performance deep learning library,” in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019, pp. 8024–8035.

[17] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, Anchorage, AK, USA, 2019, pp. 2623–2631. doi: 10.1145/3292500.3330701.

[18] S. S. Shapiro and M. B. Wilk, “An analysis of variance test for normality (complete samples),” *Biometrika*, vol. 52, no. 3–4, pp. 591–611, 1965. doi: 10.1093/biomet/52.3-4.591.

[19] H. Levene, “Robust tests for equality of variances,” in *Contributions to Probability and Statistics: Essays in Honor of Harold Hotelling*, I. Olkin, Ed. Stanford, CA, USA: Stanford Univ. Press, 1960, pp. 278–292.

[20] F. Wilcoxon, “Individual comparisons by ranking methods,” *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945. doi: 10.2307/3001968.

[21] H. B. Mann and D. R. Whitney, “On a test of whether one of two random variables is stochastically larger than the other,” *The Annals of Mathematical Statistics*, vol. 18, no. 1, pp. 50–60, 1947. doi: 10.1214/aoms/1177730491.

[22] B. L. Welch, “The generalization of ‘Student’s’ problem when several different population variances are involved,” *Biometrika*, vol. 34, no. 1–2, pp. 28–35, 1947. doi: 10.1093/biomet/34.1-2.28.

[23] S. Holm, “A simple sequentially rejective multiple test procedure,” *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979.

[24] Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: A practical and powerful approach to multiple testing,” *Journal of the Royal Statistical Society: Series B*, vol. 57, no. 1, pp. 289–300, 1995. doi: 10.1111/j.2517-6161.1995.tb02031.x.

[25] J. P. Simmons, L. D. Nelson, and U. Simonsohn, “False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant,” *Psychological Science*, vol. 22, no. 11, pp. 1359–1366, 2011. doi: 10.1177/0956797611417632.



# **Động lực học của Sự suy giảm: Phân tích Lý thuyết và Thực nghiệm về Tốc độ Hội tụ trong Tối ưu hóa Phi lồi**

## **Phần I: Bối cảnh của Tối ưu hóa Phi lồi**

Phần này thiết lập bối cảnh và những thách thức cơ bản của vấn đề nghiên cứu. Nó đi từ bài toán tổng quát về tối ưu hóa trong học sâu đến những hạn chế cụ thể của các thuật toán cơ bản nhất, qua đó tạo tiền đề cho sự cần thiết của các phương pháp tiên tiến hơn được thảo luận trong Phần II.

### **1\. Thách thức Tối ưu hóa trong Học sâu**

#### **1.1. Vai trò trung tâm của Tối ưu hóa**

Tối ưu hóa là động cơ cốt lõi của quá trình huấn luyện các mạng nơ-ron sâu (Deep Neural Networks \- DNNs). Quá trình huấn luyện về bản chất là một bài toán tối ưu hóa số quy mô lớn: tìm kiếm một tập hợp các tham số mô hình $\\theta$ trong một không gian có số chiều cực lớn nhằm cực tiểu hóa một hàm mất mát $f(\\theta)$.1 Quy mô của thách thức này rất lớn, với các mô hình hiện đại có thể chứa hàng triệu, thậm chí hàng tỷ tham số, đòi hỏi các thuật toán không chỉ hiệu quả về mặt lý thuyết mà còn phải khả thi về mặt tính toán.1

#### **1.2. Bản chất Phi lồi của Bề mặt Mất mát**

Thách thức khoa học nền tảng phát sinh từ bản chất phi lồi (non-convex) của các bề mặt mất mát trong học sâu.1 Điều này tạo ra một sự khác biệt sâu sắc so với lĩnh vực tối ưu hóa lồi đã được nghiên cứu kỹ lưỡng, nơi mọi điểm cực tiểu cục bộ cũng là điểm cực tiểu toàn cục và các thuật toán dựa trên gradient được đảm bảo sẽ hội tụ đến nghiệm tối ưu toàn cục.1 Ngược lại, các bề mặt mất mát của DNNs có cấu trúc topo vô cùng phức tạp, đặc trưng bởi các yếu tố sau:

* **Vô số điểm cực tiểu cục bộ:** Ban đầu, đây được coi là một trở ngại lớn, vì người ta lo ngại rằng các thuật toán có thể bị "mắc kẹt" trong các điểm cực tiểu cục bộ có chất lượng kém. Tuy nhiên, các nghiên cứu sau này cho thấy rằng trong các mạng lớn, hầu hết các điểm cực tiểu cục bộ có giá trị hàm mất mát tương đương nhau và thường cho hiệu suất tổng quát hóa tốt.1 Một số công trình còn chỉ ra rằng các điểm cực tiểu này thường được kết nối với nhau trong không gian tham số, tạo thành các "thung lũng" rộng lớn.1  
* **Các vùng cao nguyên (Plateaus) rộng lớn:** Đây là những vùng trong không gian tham số nơi bề mặt mất mát gần như bằng phẳng, dẫn đến gradient có giá trị rất nhỏ.1 Khi một thuật toán tối ưu hóa đi vào một vùng cao nguyên, tốc độ hội tụ của nó có thể giảm đi đáng kể, gần như dừng lại, ngay cả khi nó còn cách xa điểm cực tiểu.  
* **Sự phổ biến của các điểm yên ngựa (Saddle Points):** Đây được xác định là thách thức chủ đạo trong tối ưu hóa phi lồi chiều cao.1 Một điểm yên ngựa là một điểm dừng bậc một (first-order stationary point), nơi gradient bằng không ($\\nabla f(\\theta) \= 0$), nhưng nó không phải là điểm cực tiểu cục bộ. Tại điểm này, bề mặt cong lên ở một số hướng và cong xuống ở các hướng khác. Nghiên cứu của Dauphin và cộng sự đã chỉ ra rằng trong không gian nhiều chiều, tỷ lệ số điểm yên ngựa so với số điểm cực tiểu cục bộ tăng theo cấp số nhân với số chiều.1 Do đó, một thuật toán dựa trên gradient có nhiều khả năng gặp phải điểm yên ngựa hơn là điểm cực tiểu cục bộ. Việc bị chậm lại hoặc mắc kẹt gần các điểm yên ngựa là một nguyên nhân chính gây ra sự hội tụ chậm trong thực tế.

#### **1.3. Trực quan hóa Thách thức**

Bản chất trừu tượng của các bề mặt mất mát chiều cao có thể được làm sáng tỏ phần nào thông qua các kỹ thuật trực quan hóa. Công trình của Li và cộng sự đã giới thiệu các phương pháp để tạo ra các lát cắt 1D và 2D có ý nghĩa của bề mặt mất mát, cho phép chúng ta "nhìn thấy" hình dạng của nó xung quanh một điểm cực tiểu.1 Những hình ảnh trực quan này đã củng cố một nhận thức quan trọng: *hình học* của bề mặt mất mát—độ cong, độ trơn, và "hình dạng" của các thung lũng—có mối liên hệ nội tại với cả khả năng huấn luyện của mô hình và hiệu suất tổng quát hóa cuối cùng của nó. Ví dụ, các điểm cực tiểu nằm trong các "thung lũng" rộng và phẳng thường có khả năng tổng quát hóa tốt hơn so với các điểm cực tiểu nằm trong các "khe" hẹp và sắc.3

Từ những phân tích này, mục tiêu của một thuật toán tối ưu hóa hiện đại được định hình lại. Nó không chỉ đơn thuần là "tìm một điểm cực tiểu", mà là "duy trì sự tiến triển hiệu quả trong một địa hình phức tạp, chiều cao". Cụ thể, thuật toán phải có khả năng điều hướng qua các vùng cao nguyên rộng lớn và đặc biệt là thoát khỏi các điểm yên ngựa một cách hiệu quả. Do đó, tốc độ hội tụ đến một điểm dừng bậc một là một thước đo cần thiết nhưng chưa đủ; chúng ta còn phải xem xét tốc độ hội tụ đến một điểm dừng bậc hai (tức là thoát khỏi các điểm yên ngựa) để đánh giá đầy đủ hiệu quả của một thuật toán.

### **2\. Gradient Descent và Biến thể Ngẫu nhiên: Thuật toán Nền tảng**

#### **2.1. Thuật toán Gradient Descent (GD)**

Thuật toán nền tảng nhất trong tối ưu hóa là Gradient Descent (GD). Công thức cập nhật của nó rất đơn giản và trực quan:

$$\\theta\_{t+1} \= \\theta\_t \- \\alpha \\nabla f(\\theta\_t)$$

trong đó $\\theta\_t$ là vector tham số ở bước lặp $t$, và $\\alpha \> 0$ là tốc độ học (learning rate), một siêu tham số quyết định kích thước của mỗi bước cập nhật. Vector gradient $\\nabla f(\\theta\_t)$ chỉ ra hướng có độ dốc lớn nhất tại điểm $\\theta\_t$, và thuật toán di chuyển theo hướng ngược lại để giảm giá trị của hàm mất mát. Một nhược điểm lớn của GD là nó yêu cầu tính toán gradient trên toàn bộ tập dữ liệu huấn luyện để thực hiện một bước cập nhật duy nhất. Điều này trở nên bất khả thi về mặt tính toán đối với các tập dữ liệu quy mô lớn trong học máy hiện đại.6

#### **2.2. Stochastic Gradient Descent (SGD): Khả năng Mở rộng thông qua Nhiễu**

Để giải quyết vấn đề về khả năng mở rộng, Stochastic Gradient Descent (SGD) đã trở thành thuật toán tối ưu hóa chủ lực trong học sâu. Thay vì tính toán gradient trên toàn bộ dữ liệu, SGD ước tính gradient bằng cách chỉ sử dụng một tập con nhỏ, được gọi là mini-batch, tại mỗi bước lặp 1:

$$\\theta\_{t+1} \= \\theta\_t \- \\alpha \\nabla f(\\theta\_t; B\_k)$$

trong đó $B\_k$ là một mini-batch dữ liệu được lấy mẫu từ tập huấn luyện. Sự thay đổi này mang lại một sự đánh đổi cơ bản: SGD có được lợi thế khổng lồ về hiệu quả tính toán, nhưng phải trả giá bằng việc đưa phương sai (nhiễu) vào ước tính gradient.1  
Tiếng ồn này không hoàn toàn là một điều bất lợi. Nó có thể được xem như một dạng điều chuẩn (regularization) ngẫu nhiên, giúp thuật toán khám phá bề mặt mất mát và có khả năng thoát khỏi các điểm cực tiểu cục bộ "sắc" (sharp).1 Như Keskar và cộng sự đã chỉ ra, nhiễu trong SGD cho phép các phương pháp sử dụng mini-batch nhỏ tìm thấy các điểm cực tiểu "phẳng" (flat) hơn, vốn có liên quan đến khả năng tổng quát hóa tốt hơn.5 Do đó, sự chuyển đổi từ GD sang SGD không chỉ là một sự tối ưu hóa về mặt tính toán; nó còn thay đổi cơ bản động lực học của quá trình tối ưu hóa. Sự "kém hiệu quả" của SGD (do nhiễu) lại liên quan trực tiếp đến một trong những đặc tính mong muốn nhất của nó (khả năng tổng quát hóa tốt). Điều này tạo ra một sự căng thẳng cơ bản mà các phương pháp thích ứng sau này sẽ cố gắng giải quyết, thường đi kèm với những hậu quả không mong muốn.

#### **2.3. Phân tích Hội tụ Nền tảng**

Để phân tích tốc độ hội tụ của các thuật toán này một cách chặt chẽ, các nhà nghiên cứu thường dựa vào một số giả định về hàm mất mát. Giả định phổ biến nhất là tính trơn L (L-smoothness), hay còn gọi là gradient Lipschitz-liên tục.1 Một hàm $f$ được gọi là L-trơn nếu gradient của nó thỏa mãn:

$$\\|\\nabla f(x) \- \\nabla f(y)\\| \\le L \\|x \- y\\|$$

với mọi $x, y$ và một hằng số $L \> 0$. Giả định này về cơ bản giới hạn độ cong tối đa của hàm, đảm bảo rằng gradient không thay đổi quá đột ngột.  
Dưới giả định L-smoothness, tốc độ hội tụ kinh điển của GD và SGD đến một điểm dừng bậc một (nơi $\\|\\nabla f(\\theta)\\|^2 \\to 0$) thường được thiết lập là $O(1/T)$, trong đó $T$ là tổng số bước lặp.1 Tốc độ này được gọi là hội tụ dưới tuyến tính (sublinear), có nghĩa là sai số giảm chậm theo thời gian. Tốc độ này sẽ đóng vai trò là một tiêu chuẩn cơ bản để so sánh với các thuật toán tiên tiến hơn. Tuy nhiên, cần lưu ý rằng tốc độ này chỉ đảm bảo sự hội tụ đến một điểm dừng bất kỳ, không phân biệt đó là điểm cực tiểu cục bộ, điểm cực đại cục bộ hay điểm yên ngựa. Nó cũng không cung cấp bất kỳ đảm bảo nào về việc thoát khỏi các điểm yên ngựa một cách hiệu quả.

## **Phần II: Các Cơ chế Tăng tốc và Thích ứng**

Phần này phân tích hai cải tiến chính cho SGD, vốn là trọng tâm của nghiên cứu này: momentum và tốc độ học thích ứng. Mục tiêu là xây dựng một sự hiểu biết sâu sắc, có tính cơ chế về *cách* chúng hoạt động trước khi đánh giá một cách phê bình *liệu* chúng có hoạt động như mong đợi trong Phần III.

### **3\. Sức mạnh của Momentum: Làm mượt và Tăng tốc Sự suy giảm**

#### **3.1. Trực giác: Phép ẩn dụ Quả cầu nặng**

Momentum được giới thiệu một cách trực quan thông qua phép ẩn dụ của một hệ thống vật lý. Vector tham số được ví như một quả cầu nặng lăn xuống bề mặt mất mát.1 Thay vì chỉ di chuyển theo hướng của gradient tại mỗi điểm, quả cầu này tích lũy vận tốc (momentum) từ các bước trước đó. Vận tốc này giúp nó "lướt qua" các gập ghềnh nhỏ (nhiễu từ SGD) và tăng tốc khi lăn xuống các con dốc dài và nhất quán.

#### **3.2. Công thức hóa và các Biến thể**

* **Phương pháp Momentum Cổ điển (Heavy Ball):** Phương pháp momentum "quả cầu nặng" của Polyak được giới thiệu chính thức.1 Quá trình cập nhật được chia thành hai bước:  
  1. Cập nhật vận tốc: $v\_{t+1} \= \\beta v\_t \- \\alpha \\nabla f(\\theta\_t)$  
  2. Cập nhật vị trí: $\\theta\_{t+1} \= \\theta\_t \+ v\_{t+1}$  
     trong đó $v\_t$ là vector vận tốc tại bước $t$, và siêu tham số $\\beta \\in Thay vì tính toán gradient tại vị trí hiện tại $\\theta\_t$, NAG "nhìn về phía trước" bằng cách trước tiên thực hiện một bước di chuyển tạm thời theo hướng vận tốc cũ, sau đó tính toán gradient tại điểm "tương lai" đó. Công thức cập nhật có thể được viết như sau:  
  3. Cập nhật vận tốc: $v\_{t+1} \= \\beta v\_t \- \\alpha \\nabla f(\\theta\_t \+ \\beta v\_t)$  
  4. Cập nhật vị trí: $\\theta\_{t+1} \= \\theta\_t \+ v\_{t+1}$  
     Bước đi "đón đầu" này cung cấp một hiệu ứng điều chỉnh, giúp ngăn chặn việc "vọt lố" (overshooting) và cải thiện sự ổn định, đặc biệt là với các giá trị $\\beta$ cao. Công trình của Sutskever và cộng sự đã nhấn mạnh tầm quan trọng thực tiễn của nó trong học sâu.1

#### **3.3. Lợi ích của Momentum**

* **Giảm thiểu Dao động:** Trong các "khe núi" dốc hoặc các "thung lũng" hẹp (các bài toán có điều kiện xấu), các gradient liên tiếp có thể chỉ theo các hướng gần như đối lập nhau, gây ra chuyển động zig-zag không hiệu quả. Momentum tính trung bình các gradient này, làm giảm dao động theo chiều ngang của thung lũng và tăng tốc độ tiến triển dọc theo đáy thung lũng.1  
* **Tăng tốc Hội tụ:** Trong bối cảnh lồi, momentum đã được chứng minh là cải thiện tốc độ hội tụ từ $O(1/T)$ lên tốc độ tối ưu là $O(1/T^2)$. Mặc dù đảm bảo này không thể chuyển trực tiếp sang trường hợp phi lồi tổng quát, hiệu ứng tăng tốc vẫn được quan sát rộng rãi trong thực tế.

Bản chất của momentum có thể được hiểu sâu hơn từ góc độ xử lý tín hiệu. Quy tắc cập nhật vận tốc, $v\_t \= \\beta v\_{t-1} \+ (1-\\beta) g\_t$ (sử dụng một cách tham số hóa phổ biến khác), chính là định nghĩa của một trung bình động hàm mũ (Exponential Moving Average \- EMA). Điều này có nghĩa là momentum hoạt động như một bộ lọc thông thấp (low-pass filter) trên chuỗi gradient. Các dao động tần số cao trong các gradient ngẫu nhiên (do phương sai của mini-batch) sẽ bị làm mịn bởi EMA, vì các thành phần dương và âm có xu hướng triệt tiêu lẫn nhau trong các cửa sổ thời gian ngắn. Ngược lại, hướng đi bền bỉ, tiềm ẩn của gradient thực sự (tín hiệu) sẽ được củng cố và khuếch đại bởi EMA. Do đó, khả năng "giảm thiểu dao động" và "tăng tốc" của momentum là hệ quả trực tiếp của bản chất xử lý tín hiệu của nó: nó lọc nhiễu để ước tính tốt hơn hướng suy giảm thực sự, dẫn đến một quỹ đạo ổn định và trực tiếp hơn đến điểm cực tiểu.

### **4\. Adam và Mô hình Tốc độ học Thích ứng**

#### **4.1. Động lực: Tốc độ học cho từng Tham số**

Nhu cầu về các phương pháp thích ứng xuất phát từ việc một tốc độ học toàn cục duy nhất, như được sử dụng trong SGD và Momentum, là không tối ưu cho các DNN phức tạp.1 Trong một mạng nơ-ron, các tham số khác nhau có thể có độ nhạy và tần suất cập nhật khác nhau. Ví dụ, các tham số liên quan đến các đặc trưng hiếm gặp trong dữ liệu có thể cần các bước cập nhật lớn hơn để học hiệu quả, trong khi các tham số cho các đặc trưng phổ biến có thể cần các bước đi nhỏ hơn để tinh chỉnh.

#### **4.2. Cơ chế Adam: Ước tính Mô-men Thích ứng**

Cốt lõi của thuật toán Adam (Adaptive Moment Estimation) sẽ được phân tích tỉ mỉ, dựa trên bài báo gốc của Kingma và Ba.12

* Ước tính Mô-men bậc nhất (Momentum): Tương tự như momentum, Adam duy trì một trung bình động hàm mũ của gradient, $m\_t$, được gọi là mô-men bậc nhất:

  $$m\_t \= \\beta\_1 m\_{t-1} \+ (1-\\beta\_1)g\_t$$

  Đây có thể được coi là ước tính "trung bình" của các gradient.  
* Ước tính Mô-men bậc hai (Scaling): Điểm khác biệt chính là Adam cũng duy trì một trung bình động hàm mũ của gradient bình phương, $v\_t$, được gọi là mô-men bậc hai:

  $$v\_t \= \\beta\_2 v\_{t-1} \+ (1-\\beta\_2)g\_t^2$$

  Đây là ước tính "phương sai không định tâm" của các gradient.  
* Hiệu chỉnh Thiên vị (Bias Correction): Báo cáo sẽ giải thích tầm quan trọng của các bước hiệu chỉnh thiên vị cho $m\_t$ và $v\_t$. Vì $m\_0$ và $v\_0$ được khởi tạo bằng 0, các ước tính trong vài bước lặp đầu tiên sẽ bị thiên vị về phía 0\. Adam hiệu chỉnh điều này bằng cách:

  $$\\hat{m}\_t \= \\frac{m\_t}{1 \- \\beta\_1^t} \\quad \\text{và} \\quad \\hat{v}\_t \= \\frac{v\_t}{1 \- \\beta\_2^t}$$

  Điều này đặc biệt quan trọng trong giai đoạn đầu của quá trình huấn luyện.12  
* Quy tắc Cập nhật: Cập nhật tham số cuối cùng được trình bày như sau:

  $$\\theta\_{t+1} \= \\theta\_t \- \\alpha \\frac{\\hat{m}\_t}{\\sqrt{\\hat{v}\_t} \+ \\epsilon}$$

  Công thức này cho thấy tốc độ học $\\alpha$ được điều chỉnh (scale) một cách hiệu quả cho từng tham số. Cụ thể, nó được chia cho căn bậc hai của lịch sử gradient bình phương của tham số đó. Các tham số có gradient lớn hoặc nhất quán sẽ có mẫu số lớn, dẫn đến tốc độ học hiệu quả nhỏ hơn. Ngược lại, các tham số có gradient nhỏ hoặc không thường xuyên sẽ có tốc độ học hiệu quả lớn hơn.

#### **4.3. Lời hứa ban đầu của Adam**

Những lý do cho sự chấp nhận nhanh chóng và rộng rãi của Adam được tóm tắt như sau:

* Kết hợp các lợi ích của momentum (thông qua $m\_t$) và điều chỉnh tỷ lệ thích ứng (giống như RMSProp, thông qua $v\_t$).12  
* Hiệu quả về mặt tính toán và yêu cầu ít bộ nhớ.  
* Thường hội tụ nhanh hơn nhiều so với SGD về thời gian thực hoặc số lần lặp, đặc biệt là trong giai đoạn đầu của quá trình huấn luyện.1  
* Các siêu tham số ($\\beta\_1, \\beta\_2$) thường hoạt động tốt với các giá trị mặc định (0.9 và 0.999), làm cho nó có vẻ dễ tinh chỉnh hơn.1

Bước cập nhật của Adam có thể được diễn giải như một dạng tối ưu hóa tỷ lệ tín hiệu trên nhiễu (signal-to-noise ratio) cho mỗi tham số. Tử số ($m\_t$) đại diện cho tín hiệu (hướng đi nhất quán của gradient), trong khi mẫu số ($\\sqrt{v\_t}$) đại diện cho nhiễu hoặc độ lớn (sự biến thiên của gradient). Bằng cách chia tín hiệu cho nhiễu, Adam cố gắng thực hiện các bước lớn hơn cho các tham số có tín hiệu rõ ràng, nhất quán so với mức độ nhiễu của chúng. Sự điều chỉnh động này là điều cho phép Adam đạt được tiến bộ ban đầu nhanh chóng: nó nhanh chóng xác định và đi theo các tín hiệu gradient mạnh cho một số tham số nhất định trong khi thận trọng với các tham số khác. Tuy nhiên, chính cơ chế này có thể trở thành một gánh nặng ở giai đoạn sau của quá trình huấn luyện, như sẽ được thảo luận trong Phần III.

## **Phần III: Một Cái nhìn Phê bình về Sự hội tụ và Tổng quát hóa**

Phần này tạo thành cốt lõi phân tích của báo cáo. Nó vượt ra ngoài việc mô tả các thuật toán để đánh giá một cách phê bình các đảm bảo lý thuyết của chúng và, quan trọng hơn, những khác biệt rõ rệt và thường phản trực giác giữa lý thuyết và thực tế thực nghiệm.

### **5\. Đảm bảo Lý thuyết: Hội tụ, Điểm yên ngựa và Cách thoát khỏi chúng**

#### **5.1. Hội tụ đến các Điểm dừng Bậc một (FOSPs)**

Phần này sẽ so sánh một cách có hệ thống các tốc độ hội tụ lý thuyết của GD, SGD, Momentum và Adam trong bối cảnh phi lồi tổng quát. Một bảng tóm tắt sẽ được trình bày để đối chiếu các kết quả này. Nhìn chung, dưới giả định L-smoothness, các thuật toán này đều có thể được chứng minh là hội tụ đến một vùng lân cận của một điểm dừng bậc một, với tốc độ hội tụ của chuẩn gradient bình phương $\\|\\nabla f(x\_k)\\|^2$ thường là $O(1/k)$ đối với GD và chậm hơn một chút đối với các biến thể ngẫu nhiên, phụ thuộc vào phương sai của gradient và lịch trình giảm tốc độ học.1

#### **5.2. Trường hợp đặc biệt của Điều kiện Polyak-Łojasiewicz (PL)**

Điều kiện PL, được định nghĩa là $\\frac{1}{2}\\|\\nabla f(x)\\|^2 \\ge \\mu(f(x) \- f^\*)$, sẽ được giới thiệu như một giả định mạnh mẽ, yếu hơn so với tính lồi mạnh nhưng vẫn đủ để chứng minh tốc độ hội tụ tuyến tính (hay hình học) cho các phương pháp kiểu GD.1 Điều này có nghĩa là sai số $f(x\_k) \- f^\*$ giảm theo một hệ số nhỏ hơn 1 ở mỗi bước lặp, tức là $f(x\_k) \- f^\* \\le \\rho^k (f(x\_0) \- f^\*)$ với $\\rho \< 1$. Điều kiện PL ngụ ý rằng mọi điểm dừng đều là điểm cực tiểu toàn cục, do đó loại bỏ sự tồn tại của các điểm yên ngựa và các điểm cực tiểu cục bộ "xấu".1 Mặc dù điều kiện này không đúng với hầu hết các hàm mất mát DNN, nó nhấn mạnh rằng dưới các hình học bề mặt nhất định, có thể đạt được sự hội tụ nhanh hơn nhiều. Công trình của Karimi và cộng sự là tài liệu tham khảo chính cho phân tích này.1

#### **5.3. Vượt ra ngoài Bậc một: Thoát khỏi Điểm yên ngựa và Tìm kiếm SOSP**

Đây là một góc nhìn hiện đại và quan trọng.

* Khái niệm về **điểm dừng bậc hai (Second-Order Stationary Point \- SOSP)** sẽ được định nghĩa: một điểm mà tại đó $\\|\\nabla f(\\theta)\\| \\le \\epsilon$ và giá trị riêng nhỏ nhất của ma trận Hessian là $\\lambda\_{min}(\\nabla^2 f(\\theta)) \\ge \-\\sqrt{\\rho\\epsilon}$.1 Điều kiện thứ hai đảm bảo rằng không có hướng nào có độ cong âm đáng kể để có thể thoát ra.  
* Công trình đột phá của Jin và cộng sự ("How to Escape Saddle Points Efficiently") sẽ được phân tích chi tiết.2 Phát hiện quan trọng của họ—rằng thuật toán gradient descent đơn giản có thêm nhiễu (perturbed gradient descent) có thể tìm thấy một SOSP trong một khoảng thời gian chỉ phụ thuộc vào loga đa thức của số chiều—sẽ được trình bày như một kết quả mang tính bước ngoặt. Nó chứng minh rằng đối với một lớp hàm rộng, các điểm yên ngựa không phải là một rào cản thuật toán cơ bản đối với các phương pháp đơn giản.  
* Hàm ý của kết quả này đối với SGD, Momentum và Adam sẽ được thảo luận. Nhiễu vốn có trong SGD và động lượng có hướng trong các phương pháp khác được cho là cung cấp một hiệu ứng "nhiễu loạn" tương tự giúp thoát khỏi các điểm yên ngựa. Bước đột phá lý thuyết này đã chuyển trọng tâm của nghiên cứu tối ưu hóa. Nó ngụ ý rằng sự "chậm chạp" của GD/SGD không phải do bị "mắc kẹt" vĩnh viễn tại các điểm yên ngựa, mà là do thời gian cần thiết để *đi qua vùng lân cận* của một điểm yên ngựa. Điều này làm cho *động lực học* của việc thoát ra, chứ không chỉ là khả năng thoát ra, trở thành yếu tố quan trọng. Do đó, sự khác biệt chính giữa các thuật toán như SGD, Momentum và Adam không nằm ở chỗ *liệu* chúng có thoát khỏi điểm yên ngựa hay không, mà là *chúng điều hướng qua các vùng có độ cong thấp này nhanh và hiệu quả như thế nào*. Momentum, bằng cách tích lũy vận tốc, có thể "lướt" qua vùng phẳng gần điểm yên ngựa nhanh hơn. Adam, bằng cách tăng tốc độ học cho các hướng có độ lớn gradient nhỏ, cũng có khả năng tăng tốc độ thoát ra.

**Bảng 1: Tóm tắt So sánh Tốc độ Hội tụ Lý thuyết**

| Thuật toán | Giả định: Phi lồi L-trơn Tổng quát (đến FOSP) | Giả định: Lồi Mạnh / Điều kiện PL | Ghi chú về SOSP |
| :---- | :---- | :---- | :---- |
| **Gradient Descent (GD)** | Tốc độ hội tụ của $\\|\\nabla f(x\_k)\\|$ là $ $O(1/$ | Tuyến tính: $f(x\_k) \- f^\* \= O(\\rho^k)$ | Có thể mất thời gian theo cấp số nhân để thoát khỏi các điểm yên ngựa suy biến. |
| **Stochastic GD (SGD)** | Tốc độ hội tụ của $\\mathbb{E}\[\\|\\nabla f(x\_k)\\|^$ là $ $O(1/\\sqrt{k$ hoặc $ $O(1/$ tùy thuộc vào lịch trình tốc độ học và phương sai. | Cận tuyến tính (sublinear) hoặc tuyến tính tùy thuộc vào lịch trình tốc độ học. | Nhiễu vốn có giúp thoát khỏi các điểm yên ngựa không suy biến trong thời gian đa thức.1 |
| **SGD with Momentum** | Tốc độ tương tự SGD nhưng thường nhanh hơn trong thực tế do giảm dao động. Phân tích chặt chẽ phức tạp hơn. | Tăng tốc trong trường hợp lồi (tỷ lệ hội tụ tốt hơn so với GD/SGD). | Được kỳ vọng sẽ thoát khỏi điểm yên ngựa nhanh hơn SGD do vận tốc tích lũy.1 |
| **Adam / AdamW** | Phân tích phức tạp. Các công trình ban đầu có sai sót.\[18\] Các phân tích sau này cho thấy sự hội tụ dưới các giả định chặt chẽ hơn hoặc với các biến thể như AMSGrad. | Phân tích phức tạp. Không có đảm bảo hội tụ tuyến tính đơn giản như GD. | Cơ chế thích ứng có thể giúp điều hướng các vùng yên ngựa, nhưng hành vi động học có thể phức tạp.1 |

### **6\. Bí ẩn về Tổng quát hóa: Khi Nhanh hơn không phải là Tốt hơn**

#### **6.1. Quan sát Thực nghiệm: Khoảng trống Tổng quát hóa của Adam**

Phần này trình bày nghịch lý trung tâm. Nó sẽ đi sâu vào các phát hiện từ bài báo có ảnh hưởng của Wilson và cộng sự, "The Marginal Value of Adaptive Gradient Methods in Machine Learning".19

* Tuyên bố cốt lõi: Các phương pháp thích ứng như Adam, mặc dù thường hội tụ nhanh hơn trên tập huấn luyện, nhưng thường xuyên tìm thấy các giải pháp tổng quát hóa kém hơn trên dữ liệu thử nghiệm so với SGD có momentum.  
* Họ cung cấp cả bằng chứng thực nghiệm trên các mô hình tiên tiến và một bài toán phân loại tuyến tính đơn giản được xây dựng để chứng minh rằng Adam hội tụ đến một giải pháp kém trong khi SGD tìm thấy giải pháp tối ưu.

#### **6.2. Hình học của các Giải pháp: Cực tiểu Sắc và Phẳng**

Lời giải thích cho khoảng trống này sẽ được khám phá qua lăng kính của hình học bề mặt mất mát, dựa trên công trình của Keskar và cộng sự.5

* **Cực tiểu Sắc (Sharp Minima):** Đặc trưng bởi độ cong lớn (các giá trị riêng lớn của ma trận Hessian). Những thay đổi nhỏ trong dữ liệu đầu vào có thể dẫn đến những thay đổi lớn trong giá trị mất mát. Chúng có liên quan đến khả năng tổng quát hóa kém.  
* **Cực tiểu Phẳng (Flat Minima):** Nằm trong các thung lũng rộng, mở với độ cong thấp. Giá trị mất mát không nhạy cảm với những thay đổi nhỏ. Chúng có liên quan đến khả năng tổng quát hóa tốt.  
* Giả thuyết cốt lõi: Các phương pháp sử dụng batch lớn và các trình tối ưu hóa thích ứng (như Adam) có xu hướng tìm thấy các điểm cực tiểu sắc, trong khi nhiễu vốn có trong SGD với batch nhỏ cho phép nó thoát khỏi các vùng lân cận sắc và ổn định trong các vùng phẳng hơn, mạnh mẽ hơn của bề mặt mất mát.

#### **6.3. Vai trò của Kích thước Batch**

Mối liên hệ giữa kích thước batch và độ sắc nét sẽ được làm rõ. Keskar và cộng sự cho rằng các batch nhỏ tạo ra nhiễu ngăn cản sự hội tụ đến điểm cực tiểu sắc gần nhất, khuyến khích sự khám phá. Các batch lớn có ít nhiễu hơn, và do đó hoạt động giống như GD toàn batch hơn, hội tụ đến điểm cực tiểu gần nhất, có thể là một điểm sắc.5

Sự lựa chọn trình tối ưu hóa hoạt động như một *bộ điều chuẩn ngầm*, làm lệch hướng tìm kiếm về các loại giải pháp khác nhau về chất lượng ngay cả khi chúng có cùng tổn thất huấn luyện. Trong các mô hình được tham số hóa quá mức, có nhiều (thường là vô số) cài đặt tham số đạt được lỗi huấn luyện bằng không.20 Nhiệm vụ của trình tối ưu hóa không chỉ là tìm một trong những giải pháp này, mà là tìm một giải pháp "tốt". Wilson và cộng sự cho thấy rằng SGD và Adam, bắt đầu từ cùng một điểm, có thể hội tụ đến các điểm cực tiểu toàn cục hoàn toàn khác nhau với hiệu suất thử nghiệm khác nhau đáng kể.19 Keskar và cộng sự cung cấp một lời giải thích hình học: một số điểm cực tiểu là "sắc", một số khác là "phẳng", và độ phẳng tương quan với khả năng tổng quát hóa tốt.5 Kết nối hai ý tưởng này, quy tắc cập nhật của Adam, vốn tích cực điều chỉnh tốc độ học dựa trên thống kê gradient cục bộ, có nhiều khả năng "khóa vào" và nhanh chóng đi xuống điểm cực tiểu đầu tiên mà nó tìm thấy, thường là một điểm sắc. Nhiễu trong SGD ngăn cản sự hội tụ nhanh chóng này, cho phép nó "nảy xung quanh" và cuối cùng ổn định trong một vùng lân cận rộng hơn, phẳng hơn. Do đó, bản thân thuật toán, thông qua động lực học của nó, đã ngầm điều chuẩn giải pháp.

### **7\. Sai sót trong Nền tảng: Xem xét lại Sự hội tụ của Adam**

#### **7.1. Vấn đề với "Bộ nhớ Ngắn hạn"**

Phần này tập trung vào sự phê bình về chứng minh hội tụ của Adam được trình bày chi tiết trong "On the Convergence of Adam and Beyond" của Reddi và cộng sự.18

* Vấn đề cốt lõi: Trung bình động hàm mũ cho mô-men bậc hai, $v\_t$, gán trọng số giảm dần theo cấp số nhân cho các gradient cũ hơn. Trong các bối cảnh có gradient thưa thớt nhưng chứa thông tin, thông tin từ một gradient lớn, quan trọng có thể "biến mất" quá nhanh.  
* Điều này có thể khiến tốc độ học hiệu quả tăng lên một cách bất ngờ, dẫn đến thuật toán phân kỳ hoặc không hội tụ đến điểm tối ưu.

#### **7.2. Một Phản ví dụ về Sự hội tụ**

Phản ví dụ lồi đơn giản từ Reddi và cộng sự sẽ được giải thích, minh họa một kịch bản mà Adam được chứng minh là không hội tụ đến giải pháp tối ưu do vấn đề bộ nhớ ngắn hạn này.18

#### **7.3. Giải pháp: AMSGrad và "Bộ nhớ Dài hạn"**

Giải pháp được đề xuất, AMSGrad, sẽ được giới thiệu. Nó sửa đổi bản cập nhật của Adam bằng cách duy trì *giá trị lớn nhất* của tất cả các ước tính mô-men bậc hai trong quá khứ, $\\hat{v}\_t \= \\max(\\hat{v}\_{t-1}, v\_t)$, và sử dụng giá trị này trong mẫu số. Điều này đảm bảo tốc độ học không tăng và bảo tồn "bộ nhớ" về các gradient lớn trong quá khứ, khắc phục vấn đề hội tụ.

#### **7.4. Một Giải pháp Song song: Suy giảm Trọng số Tách rời (AdamW)**

Một vấn đề thực tế khác, nhưng có liên quan, với việc triển khai Adam trong hầu hết các thư viện sẽ được thảo luận, dựa trên Loshchilov & Hutter.25

* Giải thích rằng điều chuẩn L2 tiêu chuẩn, khi kết hợp với tốc độ học thích ứng của Adam, không tương đương với "suy giảm trọng số" (weight decay) thực sự. Hiệu ứng điều chuẩn trở nên bị ghép nối với độ lớn của gradient.  
* **AdamW** được trình bày như là giải pháp, tách rời bước cập nhật suy giảm trọng số khỏi bước cập nhật gradient. Điều này thường dẫn đến khả năng tổng quát hóa được cải thiện đáng kể, làm cho Adam cạnh tranh với SGD ngay cả trên các tác vụ mà trước đây nó gặp khó khăn (như phân loại hình ảnh).

Thành công thực tiễn của Adam đã che giấu những sai sót tinh vi nhưng nghiêm trọng trong nền tảng lý thuyết và cách triển khai phổ biến của nó. Các bản sửa lỗi (AMSGrad, AdamW) cho thấy rằng tối ưu hóa mạnh mẽ không chỉ đòi hỏi sự thích ứng, mà còn cần quản lý bộ nhớ cẩn thận (AMSGrad) và áp dụng đúng cách điều chuẩn (AdamW). Thực tiễn tốt nhất hiện đại hiện nay ưu tiên AdamW hơn Adam ban đầu vì lý do này.

## **Phần IV: Cái nhìn Thực nghiệm và Khuyến nghị Thực tiễn**

Phần cuối cùng này đặt toàn bộ cuộc thảo luận vào thực tế, tận dụng trực tiếp các thí nghiệm đã được lên kế hoạch của người dùng và tổng hợp tất cả các phân tích trước đó thành các kết luận có thể hành động.

### **8\. Trực quan hóa Động lực học: Một Nghiên cứu Tình huống Thực nghiệm**

#### **8.1. Thiết kế Thí nghiệm**

Phần này được xây dựng trực tiếp từ kế hoạch nghiên cứu của người dùng.1

* **Mục tiêu:** Điều tra và trực quan hóa thực nghiệm động lực học hội tụ của SGD, SGD với Momentum và Adam.  
* **Hàm kiểm tra:** Hàm Rosenbrock, $f(x,y) \= (a-x)^2 \+ b(y-x^2)^2$, sẽ được giới thiệu là môi trường thử nghiệm chính. Các đặc điểm của nó—một thung lũng parabol hẹp, dài với một điểm cực tiểu toàn cục bên trong—làm cho nó trở thành một mô hình tương tự chiều thấp tuyệt vời cho các bề mặt có điều kiện xấu thường gặp trong DNNs.1  
* **Thuật toán:** GD/SGD, SGD với Momentum, và Adam.  
* **Các chỉ số đo lường (Metrics):** Các chỉ số chính để đánh giá sẽ được xác định: giá trị mất mát theo số lần lặp, chuẩn gradient theo số lần lặp, và quỹ đạo tham số 2D được vẽ trên bản đồ đường đồng mức của hàm Rosenbrock.1

#### **8.2. Nghiên cứu Cắt giảm (Ablation Study) trên các Siêu tham số**

Phần này sẽ thực hiện khảo sát siêu tham số có hệ thống theo kế hoạch của người dùng.1

* **$\\beta$ của Momentum:** Các thí nghiệm sẽ cho thấy việc thay đổi $\\beta$ trong SGD với Momentum ảnh hưởng đến quỹ đạo như thế nào. $\\beta$ thấp sẽ giống SGD, trong khi $\\beta$ cao sẽ cho thấy các đường đi mượt mà hơn, được tăng tốc hơn nhưng cũng có nguy cơ vọt lố.  
* **$\\beta\_1$ và $\\beta\_2$ của Adam:** Đây là cốt lõi của phân tích. Báo cáo sẽ thay đổi một cách có hệ thống $\\beta\_1$ (thành phần momentum) và $\\beta\_2$ (thành phần điều chỉnh tỷ lệ) và trực quan hóa các quỹ đạo kết quả, như được minh họa trong các biểu đồ từ tài liệu của người dùng.1 Phân tích sẽ kết nối các hình ảnh trực quan này trở lại lý thuyết: $\\beta\_1$ cao giúp duy trì hướng đi, $\\beta\_2$ cao cung cấp tốc độ học ổn định, và sự mất cân bằng (ví dụ: $\\beta\_1$ thấp, $\\beta\_2$ cao) có thể dẫn đến dao động hoặc hội tụ chậm.

#### **8.3. Phân tích Kết quả**

Các quỹ đạo được trực quan hóa và các biểu đồ hội tụ sẽ được phân tích chi tiết.

* **SGD:** Có khả năng sẽ cho thấy một đường đi chậm, zig-zag xuống thung lũng.  
* **Momentum:** Sẽ cho thấy một đường đi mượt mà hơn nhiều, nhanh hơn dọc theo đáy thung lũng, chứng tỏ hiệu quả của nó trong các bối cảnh có điều kiện xấu.  
* **Adam:** Có khả năng sẽ cho thấy sự tiến bộ ban đầu rất nhanh nhưng có thể thể hiện hành vi phức tạp hơn và có khả năng kém ổn định hơn khi nó tiến gần đến điểm cực tiểu, tùy thuộc vào các tham số $\\beta$.

Các trực quan hóa chiều thấp trên các hàm kiểm tra kinh điển như Rosenbrock không chỉ mang tính minh họa; chúng là những công cụ phân tích mạnh mẽ cung cấp bằng chứng cụ thể, trực quan cho các thuộc tính lý thuyết trừu tượng của các trình tối ưu hóa (ví dụ: khả năng giảm dao động của momentum, quỹ đạo thích ứng của Adam). Chúng bắc cầu giữa các phương trình toán học và hành vi thuật toán, biến thí nghiệm từ một bài kiểm tra hiệu suất đơn giản thành một công cụ để xây dựng trực giác, qua đó xác thực lý thuyết và làm cho sự khác biệt về cơ chế giữa các thuật toán trở nên rõ ràng và ngay lập tức.

### **9\. Tổng hợp và Bộ công cụ của Người tối ưu hóa Hiện đại**

#### **9.1. Tóm tắt các Đánh đổi**

Phần này sẽ tổng hợp toàn bộ báo cáo, tóm tắt các đánh đổi cơ bản:

* **Tốc độ Hội tụ so với Chất lượng Tổng quát hóa:** Sự căng thẳng cốt lõi giữa Adam và SGD.  
* **Đơn giản so với Phức tạp:** SGD đơn giản với một siêu tham số chính ($\\alpha$), trong khi Adam có nhiều hơn ($\\alpha, \\beta\_1, \\beta\_2, \\epsilon$).  
* **Mạnh mẽ so với Độ nhạy Siêu tham số:** Mặc dù các giá trị mặc định của Adam thường hoạt động tốt, việc đạt được hiệu suất tiên tiến có thể đòi hỏi sự tinh chỉnh cẩn thận; thực tiễn tốt nhất hiện đại với AdamW thường yêu cầu tinh chỉnh tham số suy giảm trọng số riêng biệt với tốc độ học.25

#### **9.2. Khuyến nghị cho Người thực hành**

Dựa trên sự tổng hợp, các khuyến nghị có sắc thái và dựa trên bằng chứng sẽ được cung cấp.

* **Khi nào nên sử dụng SGD với Momentum:** Vẫn là một đường cơ sở mạnh mẽ, đặc biệt là trong thị giác máy tính. Nó thường có nhiều khả năng tìm thấy một giải pháp với khả năng tổng quát hóa tốt hơn, miễn là có đủ ngân sách tính toán để tinh chỉnh siêu tham số rộng rãi.  
* **Khi nào nên sử dụng Adam/AdamW:** Tuyệt vời để tạo mẫu nhanh và là lựa chọn mặc định cho nhiều bài toán (đặc biệt là trong NLP như Transformers). AdamW nên được ưu tiên hơn Adam tiêu chuẩn do suy giảm trọng số được tách rời, giúp cải thiện điều chuẩn và tổng quát hóa.25

**Bảng 2: Đặc điểm Thuật toán và các Đánh đổi Thực tiễn**

| Thuật toán | Tốc độ Hội tụ Ban đầu | Tổng quát hóa Cuối cùng (Điển hình) | Độ nhạy Siêu tham số | Sử dụng Bộ nhớ | Mạnh mẽ với Gradient Thưa | Phù hợp nhất cho |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **SGD** | Chậm | Tốt | Cao (đối với $\\alpha$) | Thấp | Kém | Các bài toán đơn giản, thiết lập đường cơ sở. |
| **SGD with Momentum** | Trung bình | Rất Tốt | Cao (đối với $\\alpha, \\beta$) | Thấp | Kém | Thị giác máy tính, khi chất lượng tổng quát hóa là tối quan trọng. |
| **Adam** | Nhanh | Trung bình \- Tốt | Trung bình (mặc định thường hoạt động) | Trung bình (lưu $m\_t, v\_t$) | Tốt | Xử lý ngôn ngữ tự nhiên (NLP), tạo mẫu nhanh. |
| **AdamW** | Nhanh | Tốt \- Rất Tốt | Trung bình (cần tinh chỉnh weight decay) | Trung bình (lưu $m\_t, v\_t$) | Tốt | Lựa chọn mặc định hiện đại cho hầu hết các tác vụ, đặc biệt là Transformers. |

#### **9.3. Hướng đi Tương lai**

Báo cáo sẽ kết thúc bằng cách đề cập ngắn gọn đến các nghiên cứu đang diễn ra trong lĩnh vực tối ưu hóa, chẳng hạn như các phương pháp bậc hai, các thuật toán thích ứng mới, và nỗ lực không ngừng để thu hẹp khoảng cách giữa lý thuyết và thực hành trong tối ưu hóa học sâu.6

#### **Works cited**

1. Đăng Ký Đề Tài NCKH.pdf  
2. How to Escape Saddle Points Efficiently \- Proceedings of Machine ..., accessed on November 3, 2025, [http://proceedings.mlr.press/v70/jin17a/jin17a.pdf](http://proceedings.mlr.press/v70/jin17a/jin17a.pdf)  
3. Visualizing the Loss Landscape of Neural Nets \- The VITALab website, accessed on November 3, 2025, [https://vitalab.github.io/article/2020/05/01/lossLandscape.html](https://vitalab.github.io/article/2020/05/01/lossLandscape.html)  
4. Visualizing the Loss Landscape of Neural Nets, accessed on November 3, 2025, [https://arxiv.org/abs/1712.09913](https://arxiv.org/abs/1712.09913)  
5. On Large-Batch Training for Deep Learning: Generalization Gap ..., accessed on November 3, 2025, [https://arxiv.org/pdf/1609.04836](https://arxiv.org/pdf/1609.04836)  
6. Optimization methods for large-scale machine learning \- Northwestern Scholars, accessed on November 3, 2025, [https://www.scholars.northwestern.edu/en/publications/optimization-methods-for-large-scale-machine-learning](https://www.scholars.northwestern.edu/en/publications/optimization-methods-for-large-scale-machine-learning)  
7. Optimization Methods for Large-Scale Machine Learning | SIAM ..., accessed on November 3, 2025, [https://epubs.siam.org/doi/10.1137/16M1080173](https://epubs.siam.org/doi/10.1137/16M1080173)  
8. ON LARGE-BATCH TRAINING FOR DEEP LEARNING: GENERALIZATION GAP AND SHARP MINIMA \- OpenReview, accessed on November 3, 2025, [https://openreview.net/pdf?id=H1oyRlYgg](https://openreview.net/pdf?id=H1oyRlYgg)  
9. Nesterov Accelerated Gradient and Momentum, accessed on November 3, 2025, [https://jlmelville.github.io/mize/nesterov.html](https://jlmelville.github.io/mize/nesterov.html)  
10. Lecture 9–10: Accelerated Gradient Descent, accessed on November 3, 2025, [https://pages.cs.wisc.edu/\~yudongchen/cs726\_sp23/Lecture\_9\_10\_accelerated\_GD.pdf](https://pages.cs.wisc.edu/~yudongchen/cs726_sp23/Lecture_9_10_accelerated_GD.pdf)  
11. adam:amethod for stochastic optimization \- arXiv, accessed on November 3, 2025, [https://arxiv.org/pdf/1412.6980](https://arxiv.org/pdf/1412.6980)  
12. Adam: A Method for Stochastic Optimization \- ResearchGate, accessed on November 3, 2025, [https://www.researchgate.net/publication/269935079\_Adam\_A\_Method\_for\_Stochastic\_Optimization](https://www.researchgate.net/publication/269935079_Adam_A_Method_for_Stochastic_Optimization)  
13. ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION, accessed on November 3, 2025, [https://ece.uwaterloo.ca/\~ece602/Projects/2018/Project7/main.html](https://ece.uwaterloo.ca/~ece602/Projects/2018/Project7/main.html)  
14. \[1412.6980v8\] Adam: A Method for Stochastic Optimization \- arXiv, accessed on November 3, 2025, [https://arxiv.org/abs/1412.6980v8?hl:ja](https://arxiv.org/abs/1412.6980v8?hl:ja)  
15. How to Escape Saddle Points Efficiently – The Berkeley Artificial Intelligence Research Blog, accessed on November 3, 2025, [https://bair.berkeley.edu/blog/2017/08/31/saddle-efficiency/](https://bair.berkeley.edu/blog/2017/08/31/saddle-efficiency/)  
16. \[1703.00887\] How to Escape Saddle Points Efficiently \- arXiv, accessed on November 3, 2025, [https://arxiv.org/abs/1703.00887](https://arxiv.org/abs/1703.00887)  
17. ON THE CONVERGENCE OF ADAM AND BEYOND \- OpenReview, accessed on November 3, 2025, [https://openreview.net/pdf?id=ryQu7f-RZ](https://openreview.net/pdf?id=ryQu7f-RZ)  
18. The Marginal Value of Adaptive Gradient Methods in Machine Learning \- Semantic Scholar, accessed on November 3, 2025, [https://www.semanticscholar.org/paper/The-Marginal-Value-of-Adaptive-Gradient-Methods-in-Wilson-Roelofs/1ecc2bd0bc6ffa0a2f466a058589c20593e3e57c](https://www.semanticscholar.org/paper/The-Marginal-Value-of-Adaptive-Gradient-Methods-in-Wilson-Roelofs/1ecc2bd0bc6ffa0a2f466a058589c20593e3e57c)  
19. The Marginal Value of Adaptive Gradient Methods in ... \- NIPS papers, accessed on November 3, 2025, [http://papers.neurips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf](http://papers.neurips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf)  
20. The Marginal Value of Adaptive Gradient Methods in Machine Learning \- People @EECS, accessed on November 3, 2025, [https://people.eecs.berkeley.edu/\~brecht/papers/17.WilEtAl.Ada.pdf](https://people.eecs.berkeley.edu/~brecht/papers/17.WilEtAl.Ada.pdf)  
21. Large Batch Training and Sharpe Minima · Reading List & Notes, accessed on November 3, 2025, [https://zizouhe.github.io/reading-notes/notes/Sharpe\_Minima\_Exp.html](https://zizouhe.github.io/reading-notes/notes/Sharpe_Minima_Exp.html)  
22. On the Convergence of Adam and Beyond \- Zhong Peixiang, accessed on November 3, 2025, [https://zhongpeixiang.github.io/on-the-convergence-of-adam-and-beyond/](https://zhongpeixiang.github.io/on-the-convergence-of-adam-and-beyond/)  
23. ON THE CONVERGENCE OF ADAM UNDER NON- UNIFORM SMOOTHNESS: SEPARABILITY FROM SGDM AND BEYOND \- OpenReview, accessed on November 3, 2025, [https://openreview.net/pdf?id=mEBSeSk49H](https://openreview.net/pdf?id=mEBSeSk49H)  
24. DECOUPLED WEIGHT DECAY REGULARIZATION \- OpenReview, accessed on November 3, 2025, [https://openreview.net/pdf/5963886abef941684ffc0cf670297e47fb1e5155.pdf](https://openreview.net/pdf/5963886abef941684ffc0cf670297e47fb1e5155.pdf)  
25. ‍♀️ Decoupled Weight Decay \- Composer \- Databricks Mosaic AI Training, accessed on November 3, 2025, [https://docs.mosaicml.com/projects/composer/en/latest/method\_cards/decoupled\_weight\_decay.html](https://docs.mosaicml.com/projects/composer/en/latest/method_cards/decoupled_weight_decay.html)  
26. Fixing Weight Decay Regularization in Adam \- OpenReview, accessed on November 3, 2025, [https://openreview.net/forum?id=rk6qdGgCZ](https://openreview.net/forum?id=rk6qdGgCZ)  
27. Optimization Methods for Large-Scale Machine Learning (Journal Article) | OSTI.GOV, accessed on November 3, 2025, [https://www.osti.gov/pages/biblio/1541717](https://www.osti.gov/pages/biblio/1541717)
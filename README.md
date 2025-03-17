

Real-Time Traffic Density Estimation 

Nguyễn Tấn Dũng*

FPT University, Hoa Lac High Tech Park, Hanoi, Viet Nam

 * e-mail: dungnthe171089@fpt.edu.vn

 Tóm tắt:  –Bài nghiên cứu này trình bày một phương pháp để ước lượng mật độ giao thông theo thời gian thực, qua việc áp dụng mô hình deep learning YOLOv8. Mục tiêu của nghiên cứu là hỗ trợ bước đầu cho việc phát triển một ứng dụng có khả năng ước lượng mật độ giao thông trên các tuyến đường khác nhau, cung cấp dữ liệu tức thời cho người dùng về tình hình giao thông hiện tại tại quãng đường đó.
           
           1. GIỚI THIỆU
            1.1. Vấn đề hiện tại và động lực
Trong bối cảnh hiện đại, việc đánh giá mật độ giao thông theo thời gian thực ngày càng trở nên quan trọng. Sự đô thị hóa nhanh chóng và số lượng phương tiện gia tăng đòi hỏi các hệ thống giám sát giao thông cung cấp dữ liệu kịp thời và chính xác. Thách thức đặt ra là làm thế nào để theo dõi liên tục và đánh giá hiệu quả mật độ giao thông mà không bị ảnh hưởng bởi các yếu tố như độ phân giải camera thấp hoặc độ trễ trong xử lý dữ liệu.
Ở Việt Nam, hiện chưa có nhiều ứng dụng nhận biết mật độ giao thông phổ biến như ở nhiều quốc gia khác, chẳng hạn như Mỹ, Nhật Bản và các nước châu Âu, nơi đã áp dụng các công nghệ tương tự để cải thiện quản lý giao thông. Vì vậy, nghiên cứu này sẽ tập trung vào việc áp dụng mô hình YOLO để xác định mật độ giao thông theo thời gian thực, nhằm khắc phục những hạn chế hiện tại và nâng cao hiệu quả trong quản lý giao thông.
                                    
 ![image](https://github.com/user-attachments/assets/0c4ad685-4594-416f-8cc3-98e525e288fa)
                                                                                                         
                  
   1.2. Các nghiên cứu
Phát hiện đối tượng là quá trình xác định vị trí và loại của các đối tượng trong hình ảnh hoặc video. Kết quả đầu ra bao gồm các khung giới hạn bao quanh mỗi đối tượng, cùng với nhãn lớp và điểm số độ tin cậy. Phương pháp này rất hữu ích trong việc xác định các đối tượng trong một cảnh mà không cần thông tin chi tiết về hình dạng. Một số mô hình hàng đầu cho tác vụ này bao gồm RCNN, SSD và YOLO. 
Hiện nay, các mô hình YOLO nổi bật hơn cả, nhờ vào khả năng cân bằng giữa tốc độ và độ chính xác, đặc biệt trong các tình huống phát hiện thời gian thực.
YOLOv8: Tiến Bước Mới Trong Phát Hiện Đối Tượng. Ra mắt vào tháng 1 năm 2023 bởi Ultralytics, YOLOv8 là sự phát triển mới nhất trong loạt mô hình AI YOLO. Được thiết kế cho các tác vụ như phân loại, phát hiện đối tượng và phân đoạn hình ảnh, YOLOv8 vượt trội hơn so với phiên bản trước, YOLOv7, cả về độ chính xác và tốc độ. Sử dụng Darknet53 làm kiến trúc cơ sở, YOLOv8 khai thác nhiều bản đồ đặc trưng hơn và các mạng nơ-ron tích chập hiệu quả hơn, mang lại giá trị mAP và fps cao hơn. YOLOv8 giới thiệu đầu phát hiện không cần điểm neo, cho phép ước lượng khung giới hạn pixel tương tự như các kỹ thuật phân đoạn hình ảnh, đồng thời tích hợp một hàm mất mát mới. 
Tổng hợp các tính năng này đạt được độ chính xác trung bình (mean Average Precision) là 53.7% trên bộ dữ liệu chuẩn COCO, khẳng định YOLOv8 là mô hình tiên tiến nhất trong phát hiện đối tượng.

                                                                                                   

                                                                                                       
                     1.3. Đóng góp
Lựa chọn mô hình YOLOv8 và Đánh giá ban đầu: Bắt đầu với việc lựa chọn mô hình YOLOv8 đã được huấn luyện sẵn, đánh giá hiệu suất ban đầu của mô hình trên tập dữ liệu COCO để phát hiện phương tiện.
Chuẩn bị Dữ liệu Chuyên biệt cho Phương tiện: Tạo và gán nhãn một tập dữ liệu dành riêng cho phương tiện nhằm tinh chỉnh khả năng phát hiện các loại phương tiện đa dạng của mô hình.
Ước lượng Mật độ Giao thông Thời gian Thực: Triển khai thuật toán để ước lượng mật độ giao thông bằng cách đếm phương tiện và phân tích cường độ giao thông theo thời gian thực trên dữ liệu video kiểm tra.
Tiến hành Fine Tuning: Huấn luyện lại mô hình với các phương tiện thường xuyên bắt gặp tại Việt Nam 

            
                       2. Dữ liệu
               2.1. Giới thiệu data
Dataset: 
Train: Thư mục này chứa tập huấn luyện, được sắp xếp thành hai thư mục con:
images: Chứa 536 hình ảnh được sử dụng để huấn luyện mô hình.
labels: Chứa các nhãn định dạng YOLOv8 tương ứng với các hình ảnh huấn luyện.
Valid: Thư mục này bao gồm tập xác thực, cũng được chia thành hai thư mục con:
images: Chứa 90 hình ảnh để xác thực mô hình.
labels: Chứa các nhãn định dạng YOLOv8 cho các hình ảnh xác thực.
Files:
data.yaml:
File YAML này chỉ định đường dẫn đến tập dữ liệu huấn luyện và xác thực. Nó cũng xác định số lượng lớp và tên của lớp, cần thiết cho việc huấn luyện và xác thực mô hình.



README.dataset.txt:
 Cung cấp tổng quan về tập dữ liệu, giấy phép (CC BY 4.0), và một liên kết đến tập dữ liệu trên Roboflow.

             3. Phương pháp
1. Khởi tạo và cấu hình môi trường
Cài đặt các biến môi trường và bộ lọc cảnh báo để tránh lỗi do thư viện, đảm bảo mã chạy mượt mà.
2. Tải và cấu hình mô hình YOLOv8
Tải mô hình YOLOv8 từ file yolov8n.pt. Đây là mô hình YOLOv8 phiên bản nhỏ, tối ưu cho việc phát hiện nhanh các đối tượng.
3. Phát hiện đối tượng trên ảnh tĩnh
Xử lý ảnh mẫu từ một file ảnh:
Đưa ảnh vào mô hình YOLO để nhận diện đối tượng, với ngưỡng tự tin (confidence threshold) là 50%.
Hiển thị ảnh đã nhận diện: Các đối tượng được đánh dấu bằng khung và tên nhãn, sau đó chuyển sang định dạng RGB để hiển thị chính xác màu sắc trên matplotlib.
4. Xử lý video để phân tích mật độ giao thông
Mở và đọc video: Sử dụng OpenCV để truy cập các khung hình từ video.
Định nghĩa vùng giám sát: Xác định hai làn đường bằng các đỉnh đa giác, giúp tách biệt các vùng giao thông.
Xử lý từng khung hình của video:
Làm tối các vùng không quan tâm.
Phát hiện đối tượng trên khung hình đã xử lý với YOLOv8.
Đếm số lượng phương tiện trong từng làn đường dựa trên vị trí của chúng.
Xác định cường độ giao thông: Tùy vào số lượng phương tiện, phân loại thành "Free-flow", "Smooth" hoặc "Heavy".
Hiển thị kết quả trên từng khung hình: Bao gồm số lượng phương tiện và mức độ tắc nghẽn của làn đường.
Lưu video: Lưu video đã xử lý vào file mới.
5. Fine-tuning (Huấn luyện lại) mô hình YOLOv8
Kiểm tra mô hình đã huấn luyện: Kiểm tra xem mô hình đã được huấn luyện và lưu trữ trước đó hay chưa.
Nếu chưa có mô hình:
Huấn luyện mô hình mới trên dữ liệu tùy chỉnh từ file data.yaml.
Lưu mô hình đã huấn luyện vào thư mục chỉ định.
Nếu đã có mô hình:
Tải lại mô hình đã huấn luyện từ file lưu trữ.
6. Dự đoán trên tập ảnh mới
Kiểm tra thư mục ảnh: Đảm bảo thư mục chứa ảnh cần kiểm tra tồn tại và có ảnh.
Thực hiện dự đoán: Duyệt qua tất cả ảnh trong thư mục, sử dụng mô hình YOLO đã huấn luyện.
Hiển thị và lưu kết quả: Hiển thị kết quả nhận diện của ảnh đầu tiên và lưu kết quả tất cả ảnh vào thư mục mặc định.
7. Giải phóng tài nguyên
Đóng tất cả các cửa sổ và giải phóng tài nguyên liên quan đến xử lý video, giúp tối ưu bộ nhớ và tránh lỗi khi chạy chương trình.


     
4. Kết luận
Nghiên cứu này đã đề xuất một phương pháp sử dụng mô hình YOLOv8 nhằm ước lượng mật độ giao thông theo thời gian thực, nhằm hỗ trợ trong việc quản lý và giám sát giao thông đô thị. Kết quả thử nghiệm cho thấy mô hình có khả năng phát hiện và phân loại các loại phương tiện phổ biến trên đường phố với độ chính xác cao và thời gian xử lý nhanh chóng. Cụ thể là:
Ứng dụng mô hình YOLOv8 giúp tối ưu hóa khả năng phát hiện đối tượng trong thời gian thực, vượt trội về độ chính xác và tốc độ so với các phiên bản trước của YOLO. Góp phần đáng kể vào việc cải thiện khả năng theo dõi mật độ giao thông một cách hiệu quả.
Công cụ đánh giá cường độ giao thông của hệ thống cho phép phân loại các tình huống giao thông từ "Free-flow" đến "Heavy", cung cấp cho người dùng dữ liệu tức thời về tình hình giao thông tại các tuyến đường, hỗ trợ trong việc lập kế hoạch và lựa chọn lộ trình di chuyển.
Tuy nhiên, vẫn còn một số hạn chế cần được khắc phục trong các nghiên cứu tiếp theo:
Độ chính xác cho các phương tiện đặc biệt: Mặc dù mô hình YOLOv8 đã được tinh chỉnh với các phương tiện phổ biến tại Việt Nam, độ chính xác vẫn có thể giảm khi gặp các loại phương tiện ít gặp hoặc có kích thước không đồng đều.
Việc mở rộng tập dữ liệu và thêm các loại phương tiện đặc thù tại Việt Nam cũng sẽ là một hướng đi nhằm tăng cường độ chính xác của mô hình khi áp dụng rộng rãi.


Reference:
1, conDENSE: : Conditional Density Estimation for Time Series Anomaly Detection. Authors: Alex Moore (Huma Therapeutics Ltd, Millbank Tower, London, UK) Davide Morelli (Huma Therapeutics Ltd & Institute of Biomedical Engineering, University of Oxford, UK)​ https://dl.acm.org/doi/10.1613/jair.1.14849
2, Data-Driven Energy and Population Estimation for Real-Time City-Wide Energy Footprinting. Authors: Peter Wei Xiaofan Jiang Piyaboon Kunakornjittirak, Thitirat https://dl.acm.org/doi/10.1145/3360322.3360847
3, City-scale vehicle tracking and traffic flow estimation using low frame-rate traffic cameras.
Authors: Peter Wei Haocong Shi Jiaying Yang Jingyi Qian Yinan Ji Xiaofan Jiang. https://dl.acm.org/doi/10.1145/3341162.3349336

![image](https://github.com/user-attachments/assets/642cb0cf-376a-42b8-9b30-4385e1bf9050)












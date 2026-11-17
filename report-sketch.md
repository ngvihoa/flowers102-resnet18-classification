# Phân loại 102 loài hoa bằng ResNet-18

> **Loại tài liệu:** Bản phác thảo báo cáo thực nghiệm  
> **Trạng thái:** Cập nhật ngày 11/10/2026 theo mã nguồn và output đã lưu trong notebook v2; chưa chạy lại huấn luyện.  
> **Notebook:** [`flowers-102-image-classification-using-resnet_v2.ipynb`](flowers-102-image-classification-using-resnet_v2.ipynb)

| Thông tin dự án | Nội dung |
|---|---|
| Người thực hiện | [Tên sinh viên / nhóm] |
| Học phần | [Tên học phần] |
| Giảng viên hướng dẫn | [Tên giảng viên] |
| Thời gian | [Ngày hoàn thiện báo cáo] |

## Tóm tắt

Báo cáo trình bày quá trình xây dựng mô hình phân loại 102 lớp hoa bằng ResNet-18 khởi tạo từ trọng số ImageNet. Mô hình được fine-tune trên bản Oxford Flowers 102 từ Kaggle, sử dụng tăng cường dữ liệu ở tập train và chuẩn hóa ImageNet nhất quán ở cả train, validation và test. Thí nghiệm dùng Adam, Cross-Entropy Loss, batch size 32 và có output huấn luyện trong 10 epoch.

Ở epoch cuối, mô hình đạt train accuracy **99,86%** và validation accuracy **97,43%**; validation accuracy cao nhất là **97,80%**, xuất hiện ở epoch 7 và 8. Bảng đánh giá test đã lưu ghi nhận accuracy **96,46%**, F1 macro **95,96%** và F1 weighted **96,49%**. Những kết quả này mô tả một mô hình ResNet-18 có augmentation trong bản v2. Các thông tin chưa đủ căn cứ để hoàn thiện báo cáo, gồm số ảnh test và giá trị learning rate, được ghi riêng ở phần giới hạn và kiểm tra tái lập.

## 1. Kiến thức chung về ResNet

ResNet (Residual Network) sử dụng kết nối tắt (*skip connections*) để học ánh xạ phần dư. Thay vì trực tiếp học ánh xạ `H(x)`, khối residual học `F(x) = H(x) - x` và tạo đầu ra `y = F(x) + x`. Khi kích thước đầu vào và đầu ra khác nhau, nhánh tắt có thể sử dụng phép chiếu để khớp kích thước. Cách tổ chức này giúp quá trình tối ưu mạng sâu thuận lợi hơn. [Bài báo ResNet](https://arxiv.org/abs/1512.03385).

Thí nghiệm sử dụng ResNet-18 với `ResNet18_Weights.DEFAULT`, tương ứng trọng số pretrained ImageNet-1K. [Tài liệu Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Lớp fully connected cuối được thay bằng `Linear(in_features, 102)` để phù hợp số lớp của bài toán. Notebook đưa toàn bộ `model_aug.parameters()` vào optimizer và không đóng băng backbone, vì vậy quá trình fine-tuning cập nhật cả các tầng trích xuất đặc trưng và lớp phân loại mới.

## 2. Giới thiệu tập dữ liệu

Oxford Flowers 102 gồm 102 lớp hoa thường gặp tại Vương quốc Anh. Ảnh có biến thiên về tỷ lệ, góc chụp và ánh sáng; một số lớp cũng có hình dạng hoặc màu sắc khá giống nhau. Những đặc điểm này tạo ra khó khăn khi phân biệt các lớp chỉ dựa trên ảnh. [Trang dữ liệu Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).

Notebook nạp dữ liệu tại `/kaggle/input/datasets/yousefmohamed20/oxford-102-flower-dataset/102 flower/flowers`, với ba thư mục `train`, `valid` và `test`. Tệp `cat_to_name.json` ánh xạ mã lớp sang tên hoa. Các số lượng dưới đây mô tả cách chia dữ liệu trong bản Kaggle đang dùng; chưa xác nhận cách chia này trùng với bộ chia chuẩn của Oxford.

| Tập | Số ảnh xác nhận từ output | Ảnh/lớp ít nhất | Ảnh/lớp nhiều nhất | Trung bình ảnh/lớp |
|---|---:|---:|---:|---:|
| Train | 6.552 | 27 | 206 | 64,24 |
| Validation | 818 | 1 | 28 | 8,02 |
| Test | Chưa xác nhận | — | — | — |

**Lưu ý về thống kê test:** cell 10 hiện khai báo `test_raw = ImageFolder(VALID_DIR)`, nên dòng mang tên “Test” trong bảng output thực chất lặp lại thống kê validation: 818 ảnh, ít nhất 1 và nhiều nhất 28 ảnh/lớp. Vì vậy báo cáo chưa dùng dòng này để mô tả test. Loader phục vụ đánh giá ở cell 24 được tạo riêng bằng `test_dataset = ImageFolder(TEST_DIR, transform=valid_transform)`; lỗi thống kê trên không trực tiếp đổi thư mục của loader đánh giá. Cần in `len(test_dataset)` và phân bố `test_dataset.targets` để điền chính xác hàng test.

Phân bố train và validation không đồng đều giữa các lớp. Do đó, báo cáo sử dụng cả Accuracy và Precision, Recall, F1 theo macro và weighted average để xem kết quả ở hai góc độ: mỗi lớp có trọng số ngang nhau và mỗi lớp được tính theo số mẫu.

![Phân bố số ảnh theo lớp trong tập train](figures/training_images_per_class.png)

*Hình 1. Phân bố số ảnh train theo chỉ số lớp — xuất từ cell 12 của notebook v2.*

![Phân bố độ phân giải ảnh](figures/image_resolution_distribution.png)

*Hình 2. Phân bố chiều rộng và chiều cao ảnh train — xuất từ cell 14. Output ghi nhận chiều rộng 500–1024 px và chiều cao 500–1168 px.*

![Một số ảnh mẫu trong Oxford Flowers 102](figures/sample_flowers.png)

*Hình 3. Tám ảnh train được chọn ngẫu nhiên cùng tên hoa và mã lớp — xuất từ cell 16.*

> Các đường dẫn `figures/...` là vị trí đặt ảnh khi hoàn thiện báo cáo; thư mục ảnh chưa có trong workspace hiện tại. Có thể lấy ảnh từ output đã lưu hoặc từ các tệp PNG do notebook xuất. Số cell trong báo cáo được đếm từ 1 và tính cả cell Markdown, không phải execution count.

## 3. Tổng quan mô hình và phương pháp thí nghiệm

### 3.1. Tiền xử lý và tăng cường dữ liệu

Ảnh train được xử lý bằng chuỗi phép biến đổi sau:

1. `RandomResizedCrop(224, scale=(0.8, 1.0))`: lấy vùng cắt ngẫu nhiên và đưa về kích thước 224 × 224 pixel.
2. `RandomHorizontalFlip(p=0.5)`: lật ngang với xác suất 0,5.
3. `RandomRotation(15)`: xoay ngẫu nhiên trong khoảng −15° đến 15°.
4. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: thay đổi độ sáng, độ tương phản và độ bão hòa.
5. `ToTensor()` và `Normalize(mean, std)` với mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

Validation và test dùng pipeline xác định: `Resize(256)`, `CenterCrop(224)`, `ToTensor()` và cùng phép chuẩn hóa ImageNet. Các phép biến đổi ngẫu nhiên chỉ áp dụng khi train, giúp mô hình tiếp xúc với nhiều biến thể của ảnh; pipeline đánh giá giữ đầu vào ổn định giữa các lần dự đoán.

Trong hình minh họa augmentation và dự đoán test, tensor đã chuẩn hóa được đưa về miền màu hiển thị bằng `image * std + mean`, sau đó giới hạn về `[0, 1]`. Bước này chỉ phục vụ hiển thị ảnh.

![Ví dụ ảnh gốc và các biến thể augmentation](figures/augmentation_examples.png)

*Hình 4. Một ảnh gốc và bảy biến thể augmentation sau khi hoàn nguyên chuẩn hóa để hiển thị — xuất từ cell 21.*

### 3.2. Cấu hình mô hình và huấn luyện

| Thành phần | Cấu hình trong notebook v2 |
|---|---|
| Backbone | ResNet-18, `ResNet18_Weights.DEFAULT` |
| Lớp phân loại | `Linear(model.fc.in_features, 102)` |
| Phạm vi fine-tuning | Toàn bộ tham số của mô hình |
| Loss khi train/validation | `nn.CrossEntropyLoss()` qua biến `loss_func_aug` |
| Optimizer | Adam |
| Learning rate | Truyền qua `LEARNING_RATE`; v2 chưa khai báo giá trị |
| Batch size | 32 ở cả train, validation và test |
| Số epoch đã lưu output | 10; lời gọi train tham chiếu `EPOCHS` nhưng v2 chưa khai báo biến này |
| DataLoader | `num_workers=2`, `pin_memory=True`; chỉ train dùng `shuffle=True` |
| Thiết bị theo output môi trường | CUDA |
| Phiên bản theo output môi trường | PyTorch `2.11.0+cu128`, Torchvision `0.26.0+cu128` |
| Theo dõi trong quá trình train | Train/validation loss và accuracy mỗi epoch |

Notebook huấn luyện một mô hình `model_aug`. Trong mỗi epoch, mô hình cập nhật tham số trên train rồi chuyển sang chế độ đánh giá để tính loss và accuracy trên validation, với gradient được tắt. Loss của cả train và validation được cộng có trọng số theo số ảnh từng batch, sau đó chia cho tổng số ảnh.

Hàm train trả về lịch sử bốn đại lượng: `train_loss`, `train_acc`, `val_loss` và `val_acc`. Mã nguồn hiện không lưu hoặc khôi phục checkpoint có validation tốt nhất; phần đánh giá test sử dụng `model_aug` sau khi kết thúc vòng huấn luyện.

## 4. Thực nghiệm

### 4.1. Diễn biến huấn luyện

Các số liệu sau được trích trực tiếp từ output huấn luyện của cell 36 trong notebook v2.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|
| 1 | 2,1635 | 60,12% | 0,8154 | 89,85% |
| 2 | 0,5136 | 93,64% | 0,3195 | 95,11% |
| 3 | 0,2180 | 97,56% | 0,2426 | 95,84% |
| 4 | 0,1114 | 99,04% | 0,1868 | 97,19% |
| 5 | 0,0675 | 99,47% | 0,1807 | 96,94% |
| 6 | 0,0511 | 99,50% | 0,1565 | 96,58% |
| 7 | 0,0293 | 99,92% | 0,1301 | 97,80% |
| 8 | 0,0238 | 99,86% | 0,1235 | 97,80% |
| 9 | 0,0175 | 99,91% | 0,1142 | 97,56% |
| 10 | 0,0171 | 99,86% | 0,1416 | 97,43% |

Mô hình cải thiện nhanh ở các epoch đầu: train accuracy tăng từ 60,12% lên 97,56% trong ba epoch, trong khi validation accuracy tăng từ 89,85% lên 95,84%. Từ epoch 4, validation accuracy dao động quanh 97%, còn train accuracy tiến gần 100%.

Validation accuracy tốt nhất là **97,80% ở epoch 7 và 8**. Validation loss thấp nhất là **0,1142 ở epoch 9**, cho thấy hai tiêu chí không nhất thiết đạt tối ưu ở cùng một epoch. Đến epoch 10, train loss tiếp tục giảm nhẹ nhưng validation loss tăng lên 0,1416 và validation accuracy giảm còn 97,43%. Diễn biến này gợi ý lợi ích của các epoch cuối đã giảm; cần nhiều lần chạy hoặc cơ chế chọn checkpoint để xác định điểm dừng phù hợp.

Ở epoch cuối, chênh lệch train và validation accuracy là 2,43 điểm phần trăm. Khi đọc chênh lệch này cần lưu ý train được tính trong lúc cập nhật tham số với augmentation ngẫu nhiên, còn validation dùng mô hình ở chế độ đánh giá và tiền xử lý xác định.

![Train và validation loss qua các epoch](figures/training_validation_loss.png)

*Hình 5. Train loss và validation loss của mô hình có augmentation qua 10 epoch — xuất từ cell 38.*

![Train và validation accuracy qua các epoch](figures/training_validation_accuracy.png)

*Hình 6. Train accuracy và validation accuracy của mô hình có augmentation qua 10 epoch — xuất từ cell 40.*

### 4.2. Phân tích dự đoán trên test set

Notebook đưa ảnh từ `test_loader` qua mô hình ở chế độ `eval()` và sử dụng `argmax` để chọn lớp dự đoán. Confusion matrix tổng hợp nhãn thật và nhãn dự đoán trên test; trục ngang biểu diễn lớp dự đoán, trục dọc biểu diễn lớp thật. Các phần tử trên đường chéo là dự đoán đúng, còn phần tử ngoài đường chéo thể hiện các cặp lớp bị nhầm.

![Confusion matrix của ResNet-18 có augmentation](figures/confusion_matrix_augmented_model.png)

*Hình 7. Confusion matrix trên test set — xuất từ cell 42.*

Phần minh họa tiếp theo chọn ngẫu nhiên tối đa sáu ảnh từ `test_dataset`, hiển thị tên lớp thật và tên lớp dự đoán. Viền xanh biểu thị dự đoán đúng, viền đỏ biểu thị dự đoán sai. Do ảnh được chọn ngẫu nhiên, hình minh họa có thể không chứa trường hợp sai và không đại diện đầy đủ cho các lỗi của mô hình.

![Ví dụ dự đoán ngẫu nhiên trên test set](figures/random_test_predictions.png)

*Hình 8. Các ảnh test được chọn ngẫu nhiên cùng nhãn thật và nhãn dự đoán — xuất từ cell 44.*

### 4.3. Các tiêu chí đánh giá

Accuracy là tỷ lệ dự đoán đúng trên tổng số ảnh. Precision của một lớp cho biết tỷ lệ đúng trong các ảnh được dự đoán thuộc lớp đó; Recall cho biết tỷ lệ ảnh thật của lớp được nhận diện đúng. F1 là trung bình điều hòa của Precision và Recall.

Macro average tính trung bình metric của các lớp với trọng số ngang nhau, giúp xem hiệu năng khi không để lớp nhiều ảnh chi phối. Weighted average dùng số ảnh thật của từng lớp làm trọng số. Notebook tính Precision, Recall và F1 bằng scikit-learn với `zero_division=0`; các trường hợp có mẫu số bằng 0 được gán giá trị 0.

## 5. Đánh giá

### 5.1. Metric trên test set

Các giá trị dưới đây lấy từ bảng HTML đã lưu ở cell 46. Hai dòng thể hiện hai cách tổng hợp Precision, Recall và F1 của cùng một mô hình trên cùng tập test. Loss và Accuracy là các đại lượng chung, được lặp lại trong bảng.

| Cách tổng hợp | Loss | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Macro average | 0,1531 | 96,46% | 96,10% | 96,73% | 95,96% |
| Weighted average | 0,1531 | 96,46% | 97,09% | 96,46% | 96,49% |

**Lưu ý về Loss:** giá trị 0,1531 là output đã lưu. Hàm `collect_predictions()` ở cell 44 tính loss qua biến `loss_func`, nhưng bản v2 chỉ khai báo `loss_func_aug` và không có định nghĩa `loss_func`. Vì vậy chưa xác nhận được loss function thực tế tạo ra giá trị test loss này nếu chỉ dựa trên mã nguồn v2; cần thống nhất biến và chạy lại trước khi chốt báo cáo.

### 5.2. Nhận xét kết quả

Accuracy 96,46% cho thấy mô hình phân loại đúng phần lớn ảnh trong tập test của lần chạy đã lưu. F1 macro đạt 95,96%, thấp hơn F1 weighted 0,53 điểm phần trăm. Chênh lệch này phản ánh việc thay đổi trọng số các lớp làm thay đổi kết quả tổng hợp; cần metric từng lớp và số mẫu test để xác định cụ thể lớp nào còn khó phân loại. Weighted recall bằng accuracy 96,46%, phù hợp với bài toán phân loại một nhãn trên cùng tập mẫu.

Test accuracy thấp hơn validation accuracy ở epoch cuối 0,97 điểm phần trăm. Hai giá trị khá gần nhau trong lần chạy này, nhưng chưa đủ để suy rộng sang ảnh ngoài bộ dữ liệu hoặc kết luận mô hình ổn định qua nhiều lần huấn luyện.

Thí nghiệm v2 ghi nhận hiệu năng của ResNet-18 pretrained được fine-tune với augmentation. Để định lượng riêng tác động của augmentation, cần thêm một thí nghiệm đối chứng dùng cùng cách chia dữ liệu, chuẩn hóa, cấu hình huấn luyện và quy tắc chọn checkpoint.

### 5.3. Giới hạn và các điểm cần hoàn thiện trước khi chốt báo cáo

- **Thống kê dữ liệu test:** sửa nguồn đọc của `test_raw` từ `VALID_DIR` sang `TEST_DIR`, sau đó cập nhật số ảnh và phân bố test. Cần kiểm tra ánh xạ `class_to_idx` giữa train, validation và test trước khi tái lập đánh giá.
- **Cấu hình còn thiếu:** v2 sử dụng `LEARNING_RATE` và `EPOCHS` nhưng chưa khai báo hai biến này. Output thể hiện 10 epoch; giá trị learning rate chưa có căn cứ từ v2. Cần bổ sung cấu hình để chạy notebook từ kernel mới.
- **Biến loss khi đánh giá:** thống nhất `loss_func` trong `collect_predictions()` với loss function đã khai báo và chạy lại các cell đánh giá. Output có sẵn chưa chứng minh bản v2 chạy độc lập từ đầu đến cuối.
- **Lựa chọn mô hình:** mã nguồn đánh giá mô hình cuối vòng train, chưa lưu checkpoint theo validation accuracy hoặc validation loss. Cần ghi rõ tiêu chí nếu bổ sung cơ chế chọn mô hình tốt nhất.
- **Khả năng tái lập:** notebook chưa cố định seed cho Python, NumPy và PyTorch; augmentation, shuffle và ảnh minh họa đều có yếu tố ngẫu nhiên. Chưa có kết quả nhiều lần chạy để báo cáo độ dao động.
- **Phạm vi đánh giá:** chưa có bảng metric từng lớp hoặc phân tích các cặp lớp nhầm nhiều nhất. Hình dự đoán ngẫu nhiên hỗ trợ minh họa, còn phân tích lỗi cần xem riêng những ảnh dự đoán sai.

## 6. Tài liệu tham khảo

1. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. [Bản thảo arXiv](https://arxiv.org/abs/1512.03385).
2. M.-E. Nilsback and A. Zisserman, “Automated flower classification over a large number of classes,” *Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing*, 2008. [Trang dữ liệu Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
3. Yousef Mohamed, [OxFord 102 Flower Dataset](https://www.kaggle.com/datasets/yousefmohamed20/oxford-102-flower-dataset), nguồn bản dữ liệu Kaggle được chỉ ra trong đường dẫn notebook.
4. PyTorch, [Tài liệu mô hình ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), Torchvision.
5. [`flowers-102-image-classification-using-resnet_v2.ipynb`](flowers-102-image-classification-using-resnet_v2.ipynb), mã nguồn và output thực nghiệm dùng để cập nhật báo cáo ngày 11/10/2026.

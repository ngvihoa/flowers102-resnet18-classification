
# Phân loại 102 loài hoa bằng ResNet-18

Dự án thực nghiệm **image classification** trên bản Oxford Flowers 102 được cung cấp qua Kaggle. Mô hình **ResNet-18 pretrained trên ImageNet** được thay lớp phân loại cuối và fine-tune để nhận diện 102 lớp hoa.

Repo lưu notebook với code, output huấn luyện, biểu đồ và bảng đánh giá test, cùng một bản phác thảo báo cáo tiếng Việt.

## Dữ liệu và phương pháp

- [Dataset Oxford Flowers 102 trên Kaggle](https://www.kaggle.com/datasets/yousefmohamed20/oxford-102-flower-dataset).
- Dữ liệu được đọc bằng `torchvision.datasets.ImageFolder` từ ba thư mục `train`, `valid` và `test`; `cat_to_name.json` ánh xạ mã lớp sang tên hoa.
- Output đã lưu ghi nhận **6.552 ảnh train**, **818 ảnh validation** và **102 lớp**. Cách chia này thuộc bản dữ liệu Kaggle đang dùng.
- Train sử dụng random resized crop, lật ngang, xoay ảnh và color jitter; ảnh được đưa về kích thước **224 × 224** và chuẩn hóa theo ImageNet.
- Validation và test sử dụng resize, center crop và cùng phép chuẩn hóa; mô hình được đánh giá bằng Accuracy, Precision, Recall, F1 và confusion matrix.

Notebook hiện tập trung vào **một mô hình có augmentation**, với cấu hình:

| Thành phần | Cấu hình |
| --- | --- |
| Mô hình | ResNet-18, `ResNet18_Weights.DEFAULT` |
| Fine-tuning | Toàn bộ tham số, lớp cuối có 102 đầu ra |
| Optimizer / learning rate | Adam / `1e-4` |
| Loss | Cross-Entropy |
| Batch size / số epoch | 32 / 10 |
| Thiết bị | CUDA nếu có, CPU nếu không có |

## Kết quả đã lưu

Bảng dưới đây lấy từ output đánh giá test trong [notebook hiện tại](flowers-102-image-classification-using-resnet.ipynb):

| Cách tổng hợp | Loss | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Macro average | 0,1624 | 96,21% | 96,08% | 95,44% | 95,32% |
| Weighted average | 0,1624 | 96,21% | 96,70% | 96,21% | 96,14% |

Hai dòng là hai cách tổng hợp metric của cùng mô hình; Loss và Accuracy được lặp lại. Kết quả phản ánh lần chạy đã lưu; notebook chưa cố định đầy đủ seed hoặc thực hiện nhiều lần chạy để đo độ dao động.

Phần thống kê đầu notebook hiện khai báo `test_raw = ImageFolder(VALID_DIR)`, khiến hàng “Test” lặp lại thống kê validation. **Số ảnh test chưa được xác nhận từ bảng này.** Loader đánh giá cuối được tạo riêng từ `TEST_DIR`.

## Cấu trúc repo

```text
.
├── flowers-102-image-classification-using-resnet.ipynb
├── report-sketch.md
└── README.md
```

- **Notebook:** kiểm tra dữ liệu, minh họa augmentation, fine-tune ResNet-18, vẽ đường cong học và đánh giá test. Các hình được lưu trong output notebook và được xuất vào `figures/` khi chạy các cell tương ứng.
- **[report-sketch.md](report-sketch.md):** bản phác thảo báo cáo từ một phiên bản thực nghiệm trước; còn tham chiếu notebook v2 và hình chưa có trong repo. Khi hoàn thiện báo cáo, cần đối chiếu số liệu và cấu hình với notebook hiện tại.

## Chạy notebook

### Trên Kaggle

1. Import `flowers-102-image-classification-using-resnet.ipynb` vào Kaggle.
2. Gắn dataset ở liên kết phía trên, bật GPU và cho phép truy cập Internet để tải trọng số pretrained.
3. Kiểm tra `DATASET_ROOT` và vị trí `cat_to_name.json`. Đường dẫn mặc định là:

   ```text
   /kaggle/input/datasets/yousefmohamed20/oxford-102-flower-dataset/102 flower/flowers
   ```

4. Chạy lần lượt các cell từ đầu để kiểm tra dữ liệu, huấn luyện và đánh giá.

### Trên máy cá nhân

Cài Python, PyTorch và Torchvision phù hợp với phần cứng, cùng Jupyter, matplotlib, NumPy, pandas, Pillow, scikit-learn và seaborn. Tải dữ liệu riêng, chỉnh `DATASET_ROOT` rồi mở notebook bằng Jupyter hoặc VS Code.

Môi trường của lần chạy đã lưu ghi nhận **PyTorch 2.11.0+cu128**, **Torchvision 0.26.0+cu128** và CUDA khả dụng. Notebook tự chọn CPU khi CUDA không khả dụng.

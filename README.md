# ISAS Human Activity Recognition (HAR) - Skeleton-based Action Classification

## Hướng dẫn sử dụng (Local Jupyter Notebook)

**1. Chuẩn bị môi trường**
- Cài đặt Python >= 3.8
- Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

**2. Tải dataset**
- Tải toàn bộ folder `Train_Data` (bao gồm các file keypoint, timetable, ...)
- Đảm bảo cấu trúc thư mục như sau:
```
ISAS/
├── Copy_of_ISAS_BEST(2).ipynb
├── Train_Data/
│   ├── keypoint/
│   ├── timetable/
│   └── ...
```

**3. Chạy notebook**
- Mở file `Copy_of_ISAS_BEST(2).ipynb` bằng Jupyter Notebook (local, không dùng Google Colab)
- Chạy tuần tự từ cell đầu đến cell cuối để:
  - Gắn nhãn dữ liệu keypoint cho 5 participant dựa vào timetable
  - Sinh các window nhỏ (30 frames) và lớn (120 frames) với overlap 0.5
  - Trích xuất feature cho từng window
  - Xây dựng model Extra Trees và thực hiện LOSO evaluation trên đủ 5 người

**4. Lưu ý khi chạy local**
- Đường dẫn tới dữ liệu trong notebook phải là `Train_Data/...` (không dùng đường dẫn Google Drive/Colab)
- Nếu gặp lỗi về đường dẫn, kiểm tra lại vị trí folder `Train_Data` cùng cấp với notebook

**5. Kết quả**
- Notebook sẽ tự động thực hiện toàn bộ pipeline: từ gắn nhãn, feature engineering, training, đến đánh giá LOSO cho 5 người
- Kết quả accuracy, F1-score, confusion matrix sẽ được hiển thị cuối notebook

---

**Nếu bạn cần tải lại dataset hoặc gặp vấn đề về đường dẫn, hãy liên hệ quản trị viên hoặc xem hướng dẫn chi tiết trong notebook.**

---

*Chạy notebook này là đủ để tái hiện toàn bộ quy trình xây dựng model và đánh giá LOSO trên 5 participant với dữ liệu ISAS.* 
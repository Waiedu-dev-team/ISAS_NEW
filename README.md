# ISAS Human Activity Recognition (HAR) - Skeleton-based Action Classification

## Hướng dẫn sử dụng notebook và dữ liệu

### 1. **Chạy LOCAL trên Jupyter Notebook**
- **Notebook:** `Copy_of_ISAS_BEST_(2)_Locally_versioned.ipynb`
- **Đường dẫn dữ liệu:**
  ```python
  base_path = '/workspace/isas/ISAS_NEW/'
  train_data_path = os.path.join(base_path, 'Train_Data')
  ```
- **Yêu cầu:**
  - Đặt folder `Train_Data` đúng theo đường dẫn trên (cùng cấp với notebook hoặc đúng cấu trúc)
  - Chạy tuần tự từ đầu đến cuối để thực hiện toàn bộ pipeline (gắn nhãn, feature engineering, training, LOSO evaluation)

### 2. **Chạy trên Google Colab**
- **Notebook:** `Copy_of_ISAS_BEST_(2).ipynb`
- **Đường dẫn dữ liệu:**
  ```python
  base_path = '/content/drive/MyDrive/ISAS/'
  train_data_path = os.path.join(base_path, 'Train_Data')
  ```
- **Yêu cầu:**
  - Mount Google Drive và đảm bảo dữ liệu nằm đúng vị trí trên Drive
  - Chạy notebook trên Colab để thực hiện pipeline tương tự

---

**Lưu ý:**
- Hai notebook trên là tương đương về nội dung, chỉ khác nhau về đường dẫn và môi trường chạy (local vs. Colab).
- Đảm bảo cài đặt đủ các thư viện trong `requirements.txt` trước khi chạy local.
- Nếu gặp lỗi về đường dẫn, kiểm tra lại vị trí folder `Train_Data` và cập nhật biến `base_path` cho phù hợp.

---

*Vui lòng chọn đúng notebook và cấu hình đường dẫn phù hợp với môi trường bạn sử dụng!* 
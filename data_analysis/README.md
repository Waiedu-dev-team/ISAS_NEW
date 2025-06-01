# 📊 ISAS Challenge 2025 - Complete Data Analysis Tool

Công cụ phân tích dữ liệu hoàn chỉnh cho **ISAS Challenge 2025** - Activity Recognition from Pose Keypoints

## 🎯 Giới thiệu

Tool này tích hợp tất cả chức năng phân tích dữ liệu trong **1 file duy nhất**:
- ✅ Tổng quan cấu trúc dữ liệu
- ✅ Phân tích phân phối hoạt động  
- ✅ Đánh giá chất lượng keypoints
- ✅ Tạo visualization và biểu đồ
- ✅ Export báo cáo tổng hợp

## 🚀 Sử dụng nhanh

### Chạy phân tích hoàn chỉnh (1 lệnh)

```bash
cd data_analysis
python isas_analysis_complete.py
```

**Kết quả sẽ được tạo trong thư mục `output/`:**
- `📄 comprehensive_analysis_report.md` - Báo cáo chi tiết 
- `📊 charts/activity_distribution_analysis.png` - Biểu đồ phân phối hoạt động
- `🎯 charts/keypoint_quality_analysis.png` - Biểu đồ chất lượng keypoints  
- `📈 charts/comprehensive_summary.png` - Biểu đồ tổng kết

## 📁 Cấu trúc dữ liệu yêu cầu

```
Train_Data/
├── keypoint/
│   ├── video_1.csv
│   ├── video_2.csv  
│   ├── video_3.csv
│   └── video_5.csv
├── keypointlabel/
│   ├── keypoints_with_labels_1.csv
│   ├── keypoints_with_labels_2.csv
│   ├── keypoints_with_labels_3.csv
│   └── keypoints_with_labels_5.csv
└── timetable/
    └── csv/
        ├── 1.csv
        ├── 2.csv
        ├── 3.csv
        └── 5.csv
```

## 🔍 Chi tiết phân tích

### 1. Tổng quan dữ liệu
- Kiểm tra tất cả files keypoint, label, timetable  
- Thống kê số frames, thời lượng cho mỗi user
- Đánh giá tính toàn vẹn dữ liệu

### 2. Phân tích hoạt động
- Phân loại 8 hoạt động: 4 normal, 4 abnormal
- Thống kê phân phối frames theo từng hoạt động
- Đánh giá mức độ mất cân bằng dữ liệu (normal vs abnormal)
- So sánh giữa các users

### 3. Chất lượng keypoints
- Kiểm tra missing values trong keypoints
- Phân tích độ phân giải video
- Đánh giá độ di chuyển và chất lượng tracking
- Tính điểm chất lượng tổng thể

### 4. Visualization
- **Biểu đồ phân phối hoạt động**: Bar chart, pie chart, so sánh users
- **Biểu đồ chất lượng keypoints**: Missing values, resolution, movement
- **Biểu đồ tổng kết**: Overview toàn diện của dataset

### 5. Báo cáo tổng hợp
- Thống kê chi tiết từng user và hoạt động
- Phát hiện các vấn đề cần xử lý
- Khuyến nghị preprocessing và model development
- Code examples cho các bước tiếp theo

## 📊 Kết quả mẫu

### Thống kê tổng quan:
- **Tổng frames**: ~345,000 frames (~3.2 giờ video)
- **Số users**: 4 users (User 1, 2, 3, 5)
- **Frames có nhãn**: ~216,000 frames (62.6%)
- **Hoạt động**: 8 loại (4 normal, 4 abnormal)

### Vấn đề chính:
- ⚠️ **Mất cân bằng nghiêm trọng**: 75.5% normal vs 24.5% abnormal
- ❌ **Missing labels**: 4-5% frames thiếu nhãn
- ⚠️ **Inconsistent labeling**: "Throwing" vs "Throwing things"
- ✅ **Chất lượng keypoints xuất sắc**: <0.1% missing values

## 💡 Khuyến nghị tiếp theo

### 1. Data Preprocessing:
```python
# Thống nhất labels
df['Action Label'] = df['Action Label'].replace('Throwing', 'Throwing things')

# Xử lý missing values  
df = df.dropna(subset=['Action Label'])

# Normalize keypoints
keypoint_cols = [col for col in df.columns if any(kp in col.lower() 
                for kp in ['nose', 'eye', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'])]
for col in keypoint_cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std()
```

### 2. Model Development:
- **Architecture**: LSTM/Transformer cho time series data
- **Class balancing**: `class_weight={'normal': 1, 'abnormal': 3}`  
- **Evaluation**: Leave-One-Subject-Out (LOSO) validation
- **Metrics**: F1-score (abnormal class), Overall accuracy

### 3. Feature Engineering:
```python
# Velocity features
velocity_x = np.diff(keypoint_x_columns, axis=0)
velocity_y = np.diff(keypoint_y_columns, axis=0)

# Acceleration features
acceleration_x = np.diff(velocity_x, axis=0)
acceleration_y = np.diff(velocity_y, axis=0)

# Relative positions
relative_positions = keypoints - keypoints['nose']  # Relative to nose

# Body pose angles
def calculate_angle(p1, p2, p3):
    # Calculate angle between 3 keypoints
    pass
```

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

## 📞 Hỗ trợ

Nếu gặp lỗi hoặc có câu hỏi, vui lòng:
1. Kiểm tra cấu trúc thư mục `Train_Data/` đúng format
2. Đảm bảo tất cả CSV files tồn tại
3. Kiểm tra quyền ghi trong thư mục để tạo `output/`

## 🎉 Kết luận

Tool này cung cấp phân tích toàn diện dataset ISAS Challenge 2025, giúp hiểu rõ dữ liệu trước khi phát triển model machine learning. Sử dụng kết quả phân tích để:

1. ✅ Xử lý data preprocessing đúng cách
2. ✅ Chọn architecture model phù hợp  
3. ✅ Thiết lập evaluation strategy (LOSO)
4. ✅ Optimize cho class imbalance problem

**Good luck với ISAS Challenge 2025! 🚀** 
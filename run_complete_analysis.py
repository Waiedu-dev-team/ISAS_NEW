#!/usr/bin/env python3
"""
🚀 ISAS Challenge 2025 - Complete Data Analysis Suite
Chạy toàn bộ phân tích dữ liệu một lần với đầy đủ biểu đồ và báo cáo

Author: ISAS Analysis Tool
Created: 2025-06-01
"""

import os
import sys
import time
from datetime import datetime

def print_header():
    """In header chương trình"""
    print("🚀" + "=" * 80 + "🚀")
    print("    ISAS CHALLENGE 2025 - COMPLETE DATA ANALYSIS SUITE")
    print("    Phân tích toàn diện dữ liệu với biểu đồ và báo cáo chi tiết")
    print("🚀" + "=" * 80 + "🚀")
    print(f"📅 Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_requirements():
    """Kiểm tra các yêu cầu cần thiết"""
    print("🔍 KIỂM TRA YÊU CẦU SYSTEM...")
    print("-" * 50)
    
    # Kiểm tra Python packages
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Đã cài đặt")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: Chưa cài đặt")
    
    if missing_packages:
        print(f"\n⚠️  Cần cài đặt các packages: {', '.join(missing_packages)}")
        print("   Chạy: pip install pandas numpy matplotlib seaborn")
        return False
    
    # Kiểm tra cấu trúc thư mục
    required_dirs = [
        'Train_Data/keypoint',
        'Train_Data/keypointlabel', 
        'Train_Data/timetable/csv'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            print(f"❌ Thư mục không tồn tại: {dir_path}")
        else:
            print(f"✅ Thư mục tồn tại: {dir_path}")
    
    if missing_dirs:
        print(f"\n⚠️  Thiếu thư mục dữ liệu. Kiểm tra lại cấu trúc project!")
        return False
    
    print("✅ Tất cả yêu cầu đã đáp ứng!")
    return True

def run_step(step_name, step_function, *args, **kwargs):
    """Chạy một bước phân tích với error handling"""
    print(f"\n🔄 Đang thực hiện: {step_name}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = step_function(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"✅ Hoàn thành: {step_name} ({elapsed_time:.1f}s)")
        return result, True
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Lỗi trong {step_name} ({elapsed_time:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return None, False

def run_individual_checks():
    """Chạy các kiểm tra riêng lẻ"""
    print("\n📋 CHẠY CÁC KIỂM TRA RIÊNG LẺ...")
    print("=" * 60)
    
    steps_completed = []
    steps_failed = []
    
    # Step 1: Data Overview
    try:
        from data_analysis import _1_data_overview as step1
        _, success = run_step("Kiểm tra tổng quan dữ liệu", step1.check_data_overview)
        if success:
            steps_completed.append("✅ Data Overview")
        else:
            steps_failed.append("❌ Data Overview")
    except Exception as e:
        print(f"❌ Không thể import step 1: {e}")
        steps_failed.append("❌ Data Overview")
    
    # Step 2: Activity Analysis
    try:
        from data_analysis import _2_activity_analysis as step2
        _, success1 = run_step("Phân tích hoạt động", step2.analyze_activities)
        _, success2 = run_step("Kiểm tra nhãn nhất quán", step2.check_label_consistency)
        if success1 and success2:
            steps_completed.append("✅ Activity Analysis")
        else:
            steps_failed.append("❌ Activity Analysis")
    except Exception as e:
        print(f"❌ Không thể import step 2: {e}")
        steps_failed.append("❌ Activity Analysis")
    
    # Step 3: Keypoint Analysis
    try:
        from data_analysis import _3_keypoint_analysis as step3
        _, success1 = run_step("Phân tích keypoints", step3.analyze_keypoints)
        _, success2 = run_step("Kiểm tra chất lượng", step3.check_data_quality)
        if success1 and success2:
            steps_completed.append("✅ Keypoint Analysis")
        else:
            steps_failed.append("❌ Keypoint Analysis")
    except Exception as e:
        print(f"❌ Không thể import step 3: {e}")
        steps_failed.append("❌ Keypoint Analysis")
    
    return steps_completed, steps_failed

def run_comprehensive_analysis():
    """Chạy phân tích tổng quan với biểu đồ"""
    print("\n📊 CHẠY PHÂN TÍCH TỔNG QUAN VÀ TẠO BIỂU ĐỒ...")
    print("=" * 60)
    
    try:
        from data_analysis import _5_comprehensive_analysis as comprehensive
        _, success = run_step("Phân tích tổng quan + Visualization", comprehensive.main)
        return success
    except Exception as e:
        print(f"❌ Không thể chạy comprehensive analysis: {e}")
        return False

def print_final_summary(steps_completed, steps_failed, comprehensive_success):
    """In tóm tắt cuối cùng"""
    print("\n🎉 TÓM TẮT KẾT QUỢ PHÂN TÍCH")
    print("=" * 70)
    
    print("\n📋 CÁC BƯỚC ĐÃ HOÀN THÀNH:")
    for step in steps_completed:
        print(f"   {step}")
    
    if comprehensive_success:
        print("   ✅ Comprehensive Analysis + Visualization")
    
    if steps_failed:
        print("\n❌ CÁC BƯỚC THẤT BẠI:")
        for step in steps_failed:
            print(f"   {step}")
    
    if not comprehensive_success:
        print("   ❌ Comprehensive Analysis + Visualization")
    
    # Kiểm tra files output
    print("\n📁 FILES ĐÃ TẠO:")
    
    output_files = [
        ("output/comprehensive_analysis_report.md", "📄 Báo cáo chi tiết"),
        ("output/charts/activity_distribution_analysis.png", "📊 Biểu đồ phân phối hoạt động"),
        ("output/charts/keypoint_quality_analysis.png", "🎯 Biểu đồ chất lượng keypoints"),
        ("output/charts/comprehensive_summary.png", "📈 Biểu đồ tổng kết")
    ]
    
    files_created = []
    files_missing = []
    
    for file_path, description in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            files_created.append(f"   ✅ {description}: {file_path} ({file_size/1024:.1f}KB)")
        else:
            files_missing.append(f"   ❌ {description}: {file_path}")
    
    for file_info in files_created:
        print(file_info)
    
    for file_info in files_missing:
        print(file_info)
    
    # Đánh giá tổng thể
    total_steps = len(steps_completed) + len(steps_failed)
    success_rate = len(steps_completed) / total_steps * 100 if total_steps > 0 else 0
    
    print(f"\n🏆 ĐÁNH GIÁ TỔNG THỂ:")
    print(f"   📊 Tỷ lệ thành công: {success_rate:.1f}% ({len(steps_completed)}/{total_steps} bước)")
    print(f"   📁 Files tạo được: {len(files_created)}/{len(output_files)}")
    
    if success_rate >= 75 and len(files_created) >= 3:
        print("   🎉 KẾT QUẢ: XUẤT SẮC! Phân tích hoàn thành tốt.")
    elif success_rate >= 50:
        print("   👍 KẾT QUẢ: TỐT! Hầu hết phân tích đã hoàn thành.")
    else:
        print("   ⚠️  KẾT QUẢ: CẦN CẢI THIỆN! Một số bước bị lỗi.")
    
    print(f"\n💡 BƯỚC TIẾP THEO:")
    print("   1. 📖 Xem báo cáo chi tiết: output/comprehensive_analysis_report.md")
    print("   2. 📊 Kiểm tra biểu đồ: output/charts/")
    print("   3. 🔧 Áp dụng khuyến nghị xử lý dữ liệu")
    print("   4. 🤖 Bắt đầu xây dựng mô hình ML với LOSO validation")
    
    print(f"\n⏰ Thời gian hoàn thành: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀" + "=" * 70 + "🚀")

def main():
    """Hàm chính"""
    start_time = time.time()
    
    # 1. In header
    print_header()
    
    # 2. Kiểm tra yêu cầu
    if not check_requirements():
        print("\n❌ Không đủ yêu cầu để chạy phân tích!")
        return
    
    # 3. Chạy các kiểm tra riêng lẻ
    steps_completed, steps_failed = run_individual_checks()
    
    # 4. Chạy phân tích tổng quan
    comprehensive_success = run_comprehensive_analysis()
    
    # 5. Tóm tắt kết quả
    total_time = time.time() - start_time
    print(f"\n⏱️  Tổng thời gian thực hiện: {total_time:.1f} giây ({total_time/60:.1f} phút)")
    
    print_final_summary(steps_completed, steps_failed, comprehensive_success)

if __name__ == "__main__":
    main() 
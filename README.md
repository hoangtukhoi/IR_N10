# Công cụ Tìm kiếm Phim (Movie Retrieval Engine - IR_N10)

Một công cụ tìm kiếm phim hiệu năng cao được xây dựng bằng Streamlit, triển khai các thuật toán tìm kiếm lai (Hybrid Search: BM25 + TF-IDF) và giao diện người dùng hiện đại, sinh động.

## Tính năng nổi bật
- **Tìm kiếm Lai (Hybrid Search)**: Kết hợp BM25 và TF-IDF để mang lại kết quả tìm kiếm chính xác hơn.
- **Phản hồi Giả liên quan (PRF)**: Cải thiện kết quả dựa trên các tài liệu liên quan hàng đầu.
- **Giao diện Hiện đại**: Thiết kế đáp ứng (responsive) với hiệu ứng Aurora và Glassmorphism (hiệu ứng kính mờ).
- **Bộ lọc Thông minh**: Lọc phim theo Thể loại, Năm phát hành và Điểm đánh giá.
- **Chi tiết Tương tác**: Xem thông tin chi tiết về phim bao gồm diễn viên, đạo diễn, ngân sách và doanh thu.
- **Kiểm tra Chính tả**: Gợi ý sửa lỗi cho các truy vấn tìm kiếm bị viết sai.

## Bắt đầu

### Yêu cầu hệ thống
- Python 3.8+
- Các thư viện: `pandas`, `streamlit`, `nltk`, `scikit-learn`

### Cài đặt
1. Clone repository:
   ```bash
   git clone https://github.com/hoangtukhoi/IR_N10.git
   cd IR_N10
   ```
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
   *(Lưu ý: Đảm bảo bạn có các file dữ liệu TMDB 5000 trong thư mục `data/`)*

### Chạy ứng dụng
```bash
streamlit run app.py
```

## Công nghệ sử dụng
- **Frontend**: Streamlit, Custom CSS (Glassmorphism)
- **Thuật toán**: BM25, TF-IDF, Hybrid Search, PRF
- **Xử lý dữ liệu**: Pandas, JSON
- **Xử lý ngôn ngữ tự nhiên**: NLTK (Tokenization, Preprocessing)

## Cấu trúc thư mục
- `app.py`: File chính khởi chạy ứng dụng Streamlit.
- `algorithm/`: Triển khai các thuật toán tìm kiếm (BM25, TF-IDF, Hybrid).
- `frontend/`: Các thành phần giao diện và file CSS tùy chỉnh.
- `data/`: Nơi lưu trữ bộ dữ liệu (file CSV).

## Đội ngũ thực hiện
- **Nhóm IR_N10**

---
*Dự án được thực hiện cho môn học Truy xuất thông tin (Information Retrieval).*

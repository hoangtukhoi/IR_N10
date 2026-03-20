import pandas as pd

# Đọc dữ liệu từ file CSV gốc
df = pd.read_csv('tmdb_5000_movies.csv')

# Lọc lấy 3 cột cần thiết (lưu ý: trong bộ TMDB, cột tên phim được đặt tên là 'title')
df_filtered = df[['id', 'title', 'overview']].copy()

# Đổi tên cột 'title' thành 'name'
df_filtered.rename(columns={'title': 'name'}, inplace=True)

# Xóa các dòng bị thiếu dữ liệu (NaN) ở cột overview để tránh lỗi khi xử lý NLP
df_filtered.dropna(subset=['overview'], inplace=True)

# Lưu tập dữ liệu đã lọc ra một file CSV mới
df_filtered.to_csv('movies_filtered.csv', index=False)
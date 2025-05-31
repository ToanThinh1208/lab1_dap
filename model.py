# --- START OF FILE model.py ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# In thông báo khi module này được import và chạy
print("Đang thực thi model.py: Tải dữ liệu, tiền xử lý và huấn luyện mô hình...")

df = pd.DataFrame()
model = None
features = [] # Sẽ được định nghĩa sau

try:
    # Đọc dữ liệu từ CSV
    df_raw = pd.read_csv("./googl_daily_prices.csv")
    df = df_raw.copy() # Làm việc trên bản sao

    # KIỂM TRA CỘT CẦN THIẾT
    required_cols = ['date', '1. open', '2. high', '3. low', '4. close', '5. volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Các cột sau bị thiếu trong file CSV: {', '.join(missing_cols)}")

    # Tiền xử lý dữ liệu
    df['date'] = pd.to_datetime(df['date'], errors='coerce') # Chuyển đổi ngày, lỗi sẽ thành NaT
    df = df.dropna(subset=['date']) # Xóa hàng có ngày không hợp lệ
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True) # Đảm bảo index được sắp xếp

    # Chuyển đổi các cột số liệu, lỗi sẽ thành NaN
    numeric_cols = ['1. open', '2. high', '3. low', '4. close', '5. volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Loại bỏ các hàng có giá trị thiếu trong các cột số liệu quan trọng
    df.dropna(subset=numeric_cols, inplace=True)

    if df.empty:
        raise ValueError("DataFrame rỗng sau khi làm sạch dữ liệu cơ bản (ngày, giá, volume). Kiểm tra file CSV.")

    # Tạo các đặc trưng bổ sung
    df['price_range'] = df['2. high'] - df['3. low']
    # Xử lý chia cho 0 tiềm ẩn
    df['close_open_ratio'] = np.where(df['4. close'] != 0, df['1. open'] / df['4. close'], np.nan)

    # Tạo cột volume của ngày tiếp theo (target)
    df['next_day_volume_target'] = df['5. volume'].shift(-1)

    # Loại bỏ các hàng có NaN sau khi tạo features hoặc target
    df = df.dropna()

    if df.empty:
        raise ValueError("DataFrame rỗng sau khi tạo features và target. Có thể do không đủ dữ liệu liền kề.")

    # Chọn đặc trưng và mục tiêu
    # '5. volume' của ngày hiện tại được dùng làm feature
    features = ['1. open', '2. high', '3. low', '4. close', '5. volume', 'price_range', 'close_open_ratio']
    X = df[features]
    y = df['next_day_volume_target']

    if X.empty or len(X) < 2: # Cần ít nhất 2 mẫu để split
        raise ValueError(f"Không đủ dữ liệu (sau tiền xử lý: {len(X)} mẫu) để huấn luyện mô hình.")

    # Chia dữ liệu (shuffle=False quan trọng cho chuỗi thời gian)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    if X_train.empty:
        raise ValueError("Tập huấn luyện (X_train) rỗng sau khi chia. Mô hình không thể huấn luyện.")

    # Khởi tạo và huấn luyện mô hình
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Hoàn tất huấn luyện mô hình trong model.py.")
    print(f"  R² Score trên tập test: {r2:.4f}")
    print(f"  Mean Absolute Error trên tập test: {mae:.2f}")
    print(f"  Số lượng mẫu trong df (đã xử lý): {len(df)}")
    print(f"  Phạm vi ngày trong df: {df.index.min().date()} đến {df.index.max().date()}")


except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG trong model.py khi khởi tạo: {e}")
    df = pd.DataFrame() # Đảm bảo df rỗng nếu có lỗi
    model = None        # Đảm bảo model là None nếu có lỗi

# Hàm dự đoán volume của ngày tiếp theo
def predict_next_day_volume(selected_date_input, data_df_preprocessed):
    if model is None:
        return "Lỗi: Mô hình chưa được huấn luyện thành công. Vui lòng kiểm tra log của console khi chạy app."
    if data_df_preprocessed.empty:
        return "Lỗi: Dữ liệu đã tiền xử lý (data_df_preprocessed) rỗng."

    try:
        selected_date_ts = pd.to_datetime(selected_date_input)

        # Xử lý trường hợp ngày chọn không có trong dữ liệu
        # hoặc là ngày mà không có đủ features (ví dụ ngày đầu tiên nếu features dựa trên ngày trước đó)
        if selected_date_ts not in data_df_preprocessed.index:
            # Tìm ngày làm việc gần nhất TRƯỚC ngày được chọn có trong dữ liệu
            available_dates_before = data_df_preprocessed.index[data_df_preprocessed.index < selected_date_ts]
            if available_dates_before.empty:
                return (f"Lỗi: Ngày {selected_date_ts.strftime('%Y-%m-%d')} không có trong dữ liệu "
                        f"và cũng không có dữ liệu lịch sử nào trước đó để sử dụng.")
            actual_feature_date_ts = available_dates_before.max()
            print(f"Thông tin: Ngày gốc {selected_date_ts.strftime('%Y-%m-%d')} không có. "
                  f"Sử dụng dữ liệu từ ngày gần nhất trước đó: {actual_feature_date_ts.strftime('%Y-%m-%d')}.")
        else:
            actual_feature_date_ts = selected_date_ts

        # Kiểm tra lại xem actual_feature_date_ts có trong index không (dù hơi thừa)
        if actual_feature_date_ts not in data_df_preprocessed.index:
             return f"Lỗi: Ngày được điều chỉnh {actual_feature_date_ts.strftime('%Y-%m-%d')} vẫn không tìm thấy trong dữ liệu."

        # Lấy đặc trưng của ngày được chọn (đảm bảo là DataFrame 2D)
        selected_day_features_df = data_df_preprocessed.loc[[actual_feature_date_ts], features]

        if selected_day_features_df.empty:
             return f"Lỗi: Không tìm thấy đặc trưng cho ngày {actual_feature_date_ts.strftime('%Y-%m-%d')}."
        if selected_day_features_df.isnull().values.any():
             return f"Lỗi: Dữ liệu đặc trưng cho ngày {actual_feature_date_ts.strftime('%Y-%m-%d')} chứa giá trị NaN."

        # Dự đoán volume
        predicted_volume = model.predict(selected_day_features_df)[0]
        predicted_volume = max(0, predicted_volume) # Volume không thể âm

        # Ngày mà dự đoán này áp dụng cho
        predicted_for_date_ts = actual_feature_date_ts + pd.Timedelta(days=1) # Giả sử ngày lịch tiếp theo

        actual_next_day_volume_value = None
        if predicted_for_date_ts in data_df_preprocessed.index:
            # Lấy volume thực tế của ngày được dự đoán TỪ CỘT '5. volume'
            actual_next_day_volume_value = data_df_preprocessed.loc[predicted_for_date_ts, '5. volume']
        # Hoặc, nếu muốn so sánh với target mà model học:
        # target_val = data_df_preprocessed.loc[actual_feature_date_ts, 'next_day_volume_target']

        return {
            'selected_date_for_features': actual_feature_date_ts.strftime('%Y-%m-%d'),
            'predicted_for_date': predicted_for_date_ts.strftime('%Y-%m-%d'),
            'predicted_next_day_volume': predicted_volume,
            'actual_next_day_volume': actual_next_day_volume_value
        }
    except KeyError as ke:
        return f"Lỗi Key trong dự đoán: Đặc trưng {str(ke)} không tìm thấy. Kiểm tra danh sách 'features'."
    except ValueError as ve:
        return f"Lỗi giá trị trong dự đoán: {str(ve)}"
    except Exception as e:
        return f"Lỗi không xác định trong quá trình dự đoán: {str(e)}"

if __name__ == "__main__":
    print("\nChạy model.py như một script độc lập:")
    if model is not None and not df.empty:
        if len(df.index) >= 2: # Cần ít nhất 2 ngày, ngày kế cuối để có target
            example_date_dt = df.index[-2] # Lấy ngày kế cuối từ index đã xử lý
            print(f"--- Ví dụ dự đoán sử dụng model.py ---")
            print(f"Sử dụng features từ ngày: {example_date_dt.strftime('%Y-%m-%d')}")
            result = predict_next_day_volume(example_date_dt, df)
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:,.2f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  Lỗi ví dụ: {result}")
        else:
            print("Không đủ dữ liệu trong df (sau tiền xử lý) để chạy ví dụ (cần >= 2 ngày).")
    else:
        print("Mô hình chưa được huấn luyện hoặc df trống, không thể chạy ví dụ.")
# --- END OF FILE model.py ---
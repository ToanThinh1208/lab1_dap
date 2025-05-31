# Dự án: Xây dựng Ứng dụng Web Dự báo và Chatbot Hỗ trợ với Streamlit

**Thành viên:** Nguyễn Trọng Nhân, Nguyễn Thanh Toàn Thịnh, Nguyễn Minh Phú
**Chủ đề:** Stock Market (Thị trường Chứng khoán)
**Mô tả dự án gốc:** Dự báo khối lượng giao dịch ngày mở cửa; hỗ trợ trader.

## 1. Giới thiệu Dự án

### 1.1. Mục tiêu
Xây dựng một ứng dụng web tương tác sử dụng Streamlit, bao gồm hai thành phần chính:
*   **Model Dự đoán (bên trái):** Cho phép người dùng thực hiện dự đoán (ví dụ: khối lượng giao dịch chứng khoán) dựa trên các mô hình học máy đã được huấn luyện.
*   **Chatbot Hỗ trợ (bên phải):** Tích hợp chatbot sử dụng Dialogflow để trả lời các câu hỏi liên quan đến dữ liệu, mô hình và kết quả phân tích.

### 1.2. Công nghệ sử dụng
*   **Ngôn ngữ lập trình:** Python
*   **Thư viện Machine Learning:** Scikit-learn
*   **Framework Web App:** Streamlit
*   **Nền tảng Chatbot:** Google Dialogflow
*   **Thư viện khác:** Pandas, NumPy, Matplotlib/Seaborn (cho trực quan hóa)

## 2. Xây dựng Model Dự đoán (Model for Prediction)

### 2.1. Thu thập và Tiền xử lý Dữ liệu
*   **Nguồn dữ liệu:**
    *   Mô tả chi tiết nguồn dữ liệu được sử dụng (ví dụ: dữ liệu lịch sử giao dịch từ API của sở giao dịch, file CSV từ nguồn uy tín như Yahoo Finance, Kaggle).
    *   Nêu rõ các trường dữ liệu quan trọng (Open, High, Low, Close, Volume, Date, etc.).
*   **Tiền xử lý dữ liệu:**
    *   **Làm sạch dữ liệu:** Xử lý giá trị thiếu (missing values), loại bỏ dữ liệu nhiễu hoặc không hợp lệ.
    *   **Tạo đặc trưng (Feature Engineering):**
        *   Tạo các đặc trưng mới nếu cần (ví dụ: đường trung bình động (Moving Averages), chỉ số RSI, % thay đổi giá).
        *   Chuyển đổi dữ liệu (ví dụ: chuẩn hóa, scaling).
    *   **Phân chia dữ liệu:** Chia dữ liệu thành tập huấn luyện (train) và tập kiểm thử (test). Đối với dữ liệu chuỗi thời gian, cần đảm bảo thứ tự thời gian.

### 2.2. Lựa chọn và Huấn luyện Mô hình
Sẽ huấn luyện và so sánh ít nhất 5 mô hình. Dưới đây là một số gợi ý:

1.  **Linear Regression (Hồi quy Tuyến tính):**
    *   Mô tả: Mô hình cơ bản giả định mối quan hệ tuyến tính giữa các đặc trưng đầu vào và biến mục tiêu.
    *   Ứng dụng: Dự đoán giá trị liên tục.
2.  **Random Forest Regressor (Hồi quy Rừng Ngẫu nhiên):**
    *   Mô tả: Mô hình ensemble dựa trên nhiều cây quyết định, giảm overfitting và cải thiện độ chính xác.
    *   Ứng dụng: Dự đoán giá trị liên tục, có khả năng xử lý mối quan hệ phi tuyến.
3.  **SARIMA (Seasonal Autoregressive Integrated Moving Average) - cho dữ liệu chuỗi thời gian:**
    *   Mô tả: Mô hình thống kê mạnh mẽ cho dữ liệu chuỗi thời gian có tính mùa vụ.
    *   Ứng dụng: Dự báo giá cổ phiếu, khối lượng giao dịch theo thời gian.
    *   Lưu ý: Cần xác định các tham số (p, d, q) và (P, D, Q, s).
4.  **Gradient Boosting Regressor (ví dụ: XGBoost, LightGBM, hoặc Scikit-learn GradientBoostingRegressor):**
    *   Mô tả: Mô hình ensemble xây dựng các cây tuần tự, mỗi cây cố gắng sửa lỗi của cây trước đó. Thường cho hiệu suất cao.
    *   Ứng dụng: Dự đoán giá trị liên tục, hiệu quả với dữ liệu lớn và phức tạp.
5.  **Support Vector Regressor (SVR):**
    *   Mô tả: Một biến thể của Support Vector Machines cho bài toán hồi quy, tìm một siêu phẳng tối ưu sao cho khoảng cách đến các điểm dữ liệu là nhỏ nhất trong một biên cho phép.
    *   Ứng dụng: Dự đoán giá trị liên tục, hiệu quả với không gian đặc trưng chiều cao.
6.  **(Tùy chọn thêm) K-Nearest Neighbors Regressor (KNN):**
    *   Mô tả: Dự đoán dựa trên giá trị trung bình của các điểm lân cận gần nhất.
    *   Ứng dụng: Đơn giản, dễ triển khai.

*   **Quy trình huấn luyện:**
    *   Huấn luyện từng mô hình trên tập huấn luyện.
    *   Sử dụng kỹ thuật như Cross-Validation để đánh giá độ ổn định của mô hình.
    *   Tinh chỉnh tham số (Hyperparameter Tuning) nếu cần (ví dụ: GridSearchCV, RandomizedSearchCV).

### 2.3. Đánh giá Mô hình
*   Sử dụng các chỉ số sau để đánh giá hiệu suất của các mô hình trên tập kiểm thử:
    *   **MAE (Mean Absolute Error):** `MAE = (1/n) * Σ|y_true - y_pred|`
        *   Đo lường trung bình của sai số tuyệt đối.
    *   **RMSE (Root Mean Squared Error):** `RMSE = sqrt((1/n) * Σ(y_true - y_pred)^2)`
        *   Đo lường độ lệch chuẩn của các sai số dự đoán (residuals). RMSE phạt nặng hơn các lỗi lớn.
*   Trình bày kết quả MAE và RMSE cho từng mô hình.

### 2.4. So sánh và Lựa chọn Mô hình Tốt nhất
*   Tạo bảng so sánh kết quả MAE, RMSE (và có thể các chỉ số khác như R-squared) của tất cả các mô hình.
*   Thảo luận về ưu nhược điểm của từng mô hình dựa trên kết quả và đặc điểm của dữ liệu.
*   Lựa chọn mô hình có hiệu suất tốt nhất và phù hợp nhất để triển khai trên ứng dụng Streamlit.
*   Lưu lại mô hình đã huấn luyện (ví dụ: sử dụng `joblib` hoặc `pickle`).

## 3. Tích hợp Chatbot (Chatbot Integration)

### 3.1. Thiết kế Chatbot với Dialogflow
1.  **Tạo Agent mới trong Dialogflow:**
    *   Truy cập [Dialogflow ES Console](https://dialogflow.cloud.google.com/).
    *   Tạo một Agent mới (ví dụ: `StockMarketAssistant`).
2.  **Xác định Intents (Ý định người dùng):**
    *   Tạo các Intents để xử lý các loại câu hỏi phổ biến. Ví dụ:
        *   `GetDataSources`: Hỏi về nguồn gốc dữ liệu.
            *   Training phrases: "Dữ liệu lấy từ đâu?", "Nguồn dữ liệu là gì?", "Bạn dùng data ở đâu thế?"
        *   `GetVariableMeaning`: Hỏi về ý nghĩa của một biến cụ thể.
            *   Training phrases: "Biến 'Volume' nghĩa là gì?", "Giải thích trường 'Open'?", "'Close' là gì?"
            *   Sử dụng Entities để bắt tên biến.
        *   `GetPreliminaryAnalysis`: Hỏi về kết quả phân tích sơ bộ.
            *   Training phrases: "Cho tôi biết giá trị trung bình của 'Close'?", "Số liệu thống kê cơ bản của dữ liệu?", "Khối lượng giao dịch cao nhất là bao nhiêu?"
        *   `GetModelInfo`: Hỏi về mô hình đang sử dụng.
            *   Training phrases: "Mô hình dự đoán là gì?", "Bạn dùng thuật toán nào?"
        *   `WelcomeIntent`: Chào hỏi (thường có sẵn).
        *   `FallbackIntent`: Xử lý các câu hỏi không nhận diện được (thường có sẵn).
3.  **Tạo Entities (Thực thể):**
    *   Tạo custom entities để nhận diện các thông tin cụ thể trong câu hỏi của người dùng.
        *   Ví dụ: `@variable_name` (tên các biến trong dữ liệu: Open, Close, Volume, High, Low, etc.).
        *   Ví dụ: `@analysis_type` (loại phân tích: trung bình, lớn nhất, nhỏ nhất, etc.).
4.  **Thiết kế Responses (Câu trả lời):**
    *   Trong mỗi Intent, định nghĩa các câu trả lời tĩnh hoặc sử dụng Fulfillment để tạo câu trả lời động.
5.  **Fulfillment (Tùy chọn, cho câu trả lời động):**
    *   Nếu cần câu trả lời động (ví dụ: lấy giá trị trung bình thực tế từ dữ liệu), bạn có thể kích hoạt Fulfillment cho Intent.
    *   Viết webhook (ví dụ: sử dụng Google Cloud Functions hoặc một server riêng) để xử lý logic và trả về câu trả lời cho Dialogflow.
    *   Tuy nhiên, cho chatbot "đơn giản", bạn có thể bắt đầu với các câu trả lời tĩnh được định nghĩa sẵn trong Dialogflow.
6.  **Huấn luyện và Kiểm thử:**
    *   Sau khi tạo Intents và Entities, Dialogflow sẽ tự động huấn luyện Agent.
    *   Sử dụng khung chat thử nghiệm (Try it now) trong Dialogflow Console để kiểm tra.

### 3.2. Các Câu hỏi Chatbot có thể Trả lời
*   **Nguồn dữ liệu:** "Dữ liệu bạn đang sử dụng được lấy từ đâu?"
*   **Ý nghĩa biến:**
    *   "Biến 'Open' có ý nghĩa gì trong dữ liệu chứng khoán?"
    *   "Giải thích cho tôi về trường 'Volume'."
*   **Kết quả phân tích sơ bộ:**
    *   "Giá đóng cửa trung bình là bao nhiêu?" (Có thể cần fulfillment hoặc trả lời dựa trên phân tích đã làm trước đó)
    *   "Mô tả sơ bộ về dữ liệu bạn có."
    *   "Thông tin về mô hình dự đoán bạn đang dùng?"

### 3.3. Tích hợp Chatbot vào Streamlit
1.  **Lấy thông tin xác thực từ Dialogflow:**
    *   Trong Google Cloud Console, tạo một Service Account cho project của Dialogflow Agent.
    *   Tải xuống file JSON chứa khóa xác thực (credentials).
2.  **Sử dụng Thư viện Client của Dialogflow cho Python:**
    *   Cài đặt: `pip install google-cloud-dialogflow`
    *   Viết hàm trong Python để gửi văn bản (câu hỏi của người dùng) đến Dialogflow API và nhận lại phản hồi.
    ```python
    # Ví dụ cơ bản (cần hoàn thiện với credentials và project_id, session_id)
    # import dialogflow_v2 as dialogflow # Hoặc dialogflow cho v2beta1
    #
    # def detect_intent_texts(project_id, session_id, text, language_code):
    #     session_client = dialogflow.SessionsClient() # Cần credentials
    #     session = session_client.session_path(project_id, session_id)
    #
    #     text_input = dialogflow.types.TextInput(
    #         text=text, language_code=language_code)
    #     query_input = dialogflow.types.QueryInput(text=text_input)
    #
    #     response = session_client.detect_intent(
    #         session=session, query_input=query_input)
    #
    #     return response.query_result.fulfillment_text
    ```
3.  **Tạo giao diện Chatbot trong Streamlit:**
    *   Sử dụng `st.text_input` để người dùng nhập câu hỏi.
    *   Sử dụng `st.write` hoặc các thành phần khác để hiển thị câu trả lời từ chatbot.
    *   Quản lý lịch sử hội thoại (ví dụ, lưu trong `st.session_state`).

## 4. Xây dựng Giao diện Web với Streamlit (`app.py`)

### 4.1. Bố cục Giao diện
Sử dụng `st.columns` để chia giao diện thành hai phần:
*   Cột trái: Dành cho Model Dự đoán.
*   Cột phải: Dành cho Chatbot.

```python
# app.py
import streamlit as st
# ... (import các thư viện khác: pandas, joblib, dialogflow client, etc.)

# --- Cấu hình trang ---
st.set_page_config(page_title="Dự báo Chứng khoán & Chatbot", layout="wide")

st.title("Ứng dụng Dự báo Thị trường Chứng khoán và Chatbot Hỗ trợ")

# --- Tải model đã huấn luyện ---
# model = joblib.load('path_to_your_best_model.pkl')
# scaler = joblib.load('path_to_your_scaler.pkl') # Nếu có scaler

# --- Hàm tích hợp Dialogflow ---
# def get_dialogflow_response(user_query):
#     # Gọi hàm detect_intent_texts đã định nghĩa
#     # project_id = "YOUR_DIALOGFLOW_PROJECT_ID"
#     # session_id = "unique_session_id_for_user" # có thể tạo ngẫu nhiên hoặc dựa trên session người dùng
#     # response_text = detect_intent_texts(project_id, session_id, user_query, 'vi')
#     # return response_text
#     return "Đây là câu trả lời mẫu từ chatbot cho: " + user_query # Placeholder

# --- Bố cục hai cột ---
col1, col2 = st.columns(2)

with col1:
    st.header("📈 Model Dự đoán")
    # --- 4.2. Thành phần Model Dự đoán ---
    st.subheader("Nhập thông tin để dự đoán")
    # Ví dụ:
    # feature1 = st.number_input("Đặc trưng 1 (ví dụ: Giá mở cửa hôm qua)", value=100.0)
    # feature2 = st.number_input("Đặc trưng 2 (ví dụ: Khối lượng hôm qua)", value=1000000)
    # date_to_predict = st.date_input("Chọn ngày dự đoán")

    # if st.button("Dự đoán"):
        # Xử lý input, scale nếu cần
        # input_data = pd.DataFrame([[feature1, feature2, ...]], columns=[...])
        # scaled_input = scaler.transform(input_data) # Nếu có scaler
        # prediction = model.predict(scaled_input)
        # st.success(f"Kết quả dự đoán khối lượng giao dịch: {prediction[0]:.2f}")

        # (Tùy chọn) Hiển thị thông tin mô hình
        # st.write("Mô hình sử dụng: Tên mô hình tốt nhất")
        # st.write(f"MAE: [giá trị], RMSE: [giá trị]")

with col2:
    st.header("💬 Chatbot Hỗ trợ")
    # --- 4.3. Thành phần Chatbot ---
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Hỏi chatbot:", key="chatbot_input")

    if user_input:
        # Lấy câu trả lời từ Dialogflow
        # bot_response = get_dialogflow_response(user_input)
        bot_response = "Bot: Đây là câu trả lời cho '" + user_input + "'" # Placeholder

        # Cập nhật lịch sử chat
        st.session_state.chat_history.append({"user": user_input, "bot": bot_response})

    # Hiển thị lịch sử chat
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"<div style='text-align: right; color: blue;'>Bạn: {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: left; color: green;'>Bot: {chat['bot']}</div>", unsafe_allow_html=True)
        st.markdown("---")

    if not st.session_state.chat_history:
        st.info("Chào bạn! Tôi có thể giúp gì về dữ liệu hoặc mô hình dự đoán?")
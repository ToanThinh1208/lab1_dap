import streamlit as st
# import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import model as m
from google.cloud import dialogflow_v2 as dialogflow
import uuid
# ... đầu file app.py ...
import os
import json # Thêm import này
from dotenv import load_dotenv

load_dotenv()  # tự động tìm file .env

def detect_intent_texts(project_id, session_id, text, language_code):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    try:
        # The SessionsClient will automatically use the credentials
        # from the GOOGLE_APPLICATION_CREDENTIALS environment variable
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(project_id, session_id)

        text_input = dialogflow.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.QueryInput(text=text_input)

        response = session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )

        return response.query_result.fulfillment_text
    except Exception as e:
        st.error(f"Lỗi khi kết nối tới Dialogflow: {e}")
        print(f"Dialogflow Error: {e}") # For server-side logging/debugging
        return "Xin lỗi, tôi đang gặp sự cố khi kết nối. Vui lòng thử lại sau."

# đọc data"
data = pd.read_csv("./googl_daily_prices.csv")
# --- Cấu hình trang (Luôn đặt ở đầu file) ---
st.set_page_config(
    page_title="Stock Predictor & Chatbot",
    page_icon="📈🤖",
    layout="wide", # Sử dụng layout wide để 2 cột hiển thị tốt hơn
    initial_sidebar_state="expanded"
)

st.title("Ứng dụng Dự đoán Giá Cổ phiếu & Chatbot")

# --- Chia giao diện thành 2 cột ---
col1, col2 = st.columns([3, 2]) # Tỷ lệ 3:2, cột trái rộng hơn cột phải

# --- Cột trái: Dự đoán Giá Cổ phiếu ---
with col1:
    st.header("📈 Dự đoán Giá Cổ phiếu")

    st.markdown(
        """
        <div style="background-color:#A0AEC0 ; padding:10px; border-radius:5px; margin-bottom:15px;">
            Chào mừng đến với công cụ phân tích giá cổ phiếu!
            <br>Nhập mã cổ phiếu bạn muốn xem (ví dụ: AAPL, MSFT, GOOG).
            <br>Bạn có thể chọn khoảng thời gian để xem dữ liệu lịch sử.
        </div>
        """, unsafe_allow_html=True
    )

    # Input cho mã cổ phiếu

    # Input cho mã cổ phiếu
    ticker = st.text_input("Nhập Mã Cổ phiếu (ví dụ: GOOGLE)", "GOOGLE").upper()

    # Input cho khoảng thời gian
    today = datetime.now().date()
    start_date = datetime.combine(st.date_input("Ngày bắt đầu", today - timedelta(days=365)), datetime.min.time())
    end_date = datetime.combine(st.date_input("Ngày kết thúc", today), datetime.max.time())

    # Kiểm tra ngày hợp lệ
    if start_date > end_date:
        st.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
    else:
        try:
            # xử lý ngày
            data["date"] = pd.to_datetime(data["date"])
            # Lọc dữ liệu dựa trên ngày được chọn
            filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]    

            if not filtered_data.empty:
                # Hiển thị thông tin cơ bản
                st.subheader(f"Biểu đồ Giá Cổ phiếu của {ticker}")
                st.write(f"Khối lượng giao dịch trung bình trong khoảng thời gian: {filtered_data['5. volume'].mean():,.0f} cổ phiếu")
                st.metric(label=f"Giá đóng cửa gần nhất của {ticker}", value=f"${filtered_data['4. close'].iloc[-1]:.2f}")
                
                # Tạo biểu đồ nến (Candlestick chart)
                fig = go.Figure(data=[go.Candlestick(x=filtered_data['date'],
                                                    open=filtered_data['1. open'],
                                                    high=filtered_data['2. high'],
                                                    low=filtered_data['3. low'],
                                                    close=filtered_data['4. close'],
                                                   )])
                                            

                fig.update_layout(
                    title_text=f'{ticker} Biểu đồ Giá Cổ phiếu',
                    xaxis_rangeslider_visible=False,
                    xaxis_title="Ngày",
                    yaxis_title="Giá",
                    height=500  # Chiều cao của biểu đồ
                )
                st.plotly_chart(fig, use_container_width=True)

                        # --- PHẦN DỰ ĐOÁN ---
                st.subheader("Dự đoán Khối lượng Giao dịch Ngày Tiếp Theo")
                st.markdown( # CSS
                    """
                    <style>
                    .prediction-box { /* ... CSS của bạn ... */ }
                    </style>
                    """, unsafe_allow_html=True
                )

                # Kiểm tra xem model và df từ model.py có sẵn không
                if m.model is None:
                    st.error("Mô hình dự đoán (m.model) không khả dụng. "
                            "Vui lòng kiểm tra console để xem lỗi chi tiết từ model.py.")
                elif m.df.empty: # m.df là DataFrame đã tiền xử lý trong model.py
                    st.error("Dữ liệu đã tiền xử lý (m.df) rỗng. "
                            "Vui lòng kiểm tra console để xem lỗi chi tiết từ model.py.")
                else:
                    # Date input cho dự đoán (sử dụng index của m.df)
                    min_pred_date = m.df.index.min().date()
                    max_pred_date = m.df.index.max().date()

                    # Ngày mặc định: ngày kế cuối trong m.df để có thể có actual cho ngày cuối
                    default_pred_date = m.df.index[-2].date() if len(m.df.index) >= 2 else max_pred_date

                    input_prediction_date = st.date_input(
                        "Chọn ngày để lấy dữ liệu dự đoán cho ngày tiếp theo",
                        value=default_pred_date,
                        min_value=min_pred_date,
                        max_value=max_pred_date,
                        help=(f"Dữ liệu đặc trưng sẽ được lấy từ ngày bạn chọn. "
                            f"Ngày hợp lệ trong tập đã xử lý: {min_pred_date.strftime('%d/%m/%Y')} "
                            f"đến {max_pred_date.strftime('%d/%m/%Y')}.")
                    )

                    if st.button("Thực hiện Dự đoán Volume", key="predict_button"):
                        if input_prediction_date:
                            st.write(f"Đang dự đoán cho ngày sau ngày: {input_prediction_date.strftime('%Y-%m-%d')}")
                            st.write(f"Sử dụng m.df có index từ {m.df.index.min().date()} đến {m.df.index.max().date()}")

                            predictions_result = m.predict_next_day_volume(input_prediction_date, m.df)

                            if isinstance(predictions_result, dict):
                                # ... (Phần hiển thị kết quả dự đoán giữ nguyên) ...
                                actual_volume_text = ""
                                if predictions_result.get('actual_next_day_volume') is not None:
                                    actual_volume_text = (f"<p>Volume thực tế của ngày {predictions_result['predicted_for_date']}: "
                                                        f"<b>{predictions_result['actual_next_day_volume']:,.0f}</b></p>")
                                else:
                                    actual_volume_text = f"<p>Không có dữ liệu volume thực tế cho ngày {predictions_result['predicted_for_date']} để so sánh.</p>"

                                st.markdown(
                                    f"""
                                    <div class="prediction-box" style="background-color: #A0AEC0; border: 1px solid #CBD5E0; padding: 20px; border-radius: 8px; font-size: 17px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <p style="margin-bottom: 8px;"><b style="color: #2B6CB0;">Kết quả dự đoán khối lượng giao dịch:</b></p>
                                        <p style="margin-bottom: 8px;">Sử dụng dữ liệu từ ngày: <b style="color: #2B6CB0;">{predictions_result['selected_date_for_features']}</b></p>
                                        <p style="margin-bottom: 8px;">Volume dự đoán cho ngày <b style="color: #2B6CB0;">{predictions_result['predicted_for_date']}</b>: <b style='color: green; font-size: 20px;'>{predictions_result['predicted_next_day_volume']:,.0f}</b></p>
                                        {actual_volume_text}
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                            else: # predictions_result là chuỗi lỗi
                                st.error(f"Lỗi từ hàm dự đoán: {predictions_result}")
                        else:
                            st.warning("Vui lòng chọn một ngày để dự đoán.")

            else:
                st.warning(f"Không tìm thấy dữ liệu cho mã cổ phiếu '{ticker}' trong khoảng thời gian đã chọn.")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi tải dữ liệu hoặc hiển thị: {e}")
            st.info("Vui lòng kiểm tra lại mã cổ phiếu hoặc kết nối internet của bạn.")


# --- Cột phải: Chatbot ---

with col2:
    st.header("🤖 Chatbot Hỗ trợ (Dialogflow)")

    st.markdown(
        """
        <div style="background-color:#A0AEC0 ; padding:10px; border-radius:5px; margin-bottom:15px;">
            Chào bạn! Hãy hỏi tôi bất cứ điều gì.
        </div>
        """, unsafe_allow_html=True
    )

    # Khởi tạo lịch sử tin nhắn và session_id cho Dialogflow
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Tin nhắn chào mừng ban đầu từ chatbot (có thể lấy từ Welcome Intent của Dialogflow)
        # st.session_state.messages.append({"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"})

    if "dialogflow_session_id" not in st.session_state:
        st.session_state.dialogflow_session_id = str(uuid.uuid4()) # Tạo session ID duy nhất cho mỗi người dùng/phiên

    # Hiển thị các tin nhắn cũ
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Khung nhập liệu cho chatbot
    prompt = st.chat_input("Gõ tin nhắn của bạn ở đây...")

    if prompt:
        # Thêm tin nhắn của người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Lấy phản hồi từ Dialogflow
        with st.chat_message("assistant"):
            with st.spinner("Bot đang suy nghĩ..."): # Hiệu ứng chờ
                response = detect_intent_texts(
                os.getenv("PROJECT_ID"),  # Project ID của bạn
                st.session_state.dialogflow_session_id,
                prompt,
                os.getenv("LANGUAGE_CODE") # LANGUAGE_CODE
            )
       
            st.markdown(response)
            # Thêm phản hồi của chatbot vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": response})


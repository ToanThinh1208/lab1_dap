import json
import streamlit as st

try:
    with open("./lab1-hxwy-80c276190b0a.json", "r") as f:
        data = json.load(f)
    st.success("Đã tải file JSON thành công!")
    st.json(data)
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy file JSON. Vui lòng kiểm tra đường dẫn.")
except json.JSONDecodeError:
    st.error("Lỗi: Không thể giải mã file JSON. Kiểm tra định dạng file.")
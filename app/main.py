"""Streamlit demo: load only model artifacts produced by this repository.

Run: streamlit run app/main.py
"""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from palmprint.data.dataset import resolve_path
from palmprint.data.preparation import load_manifest
from palmprint.training.classical import predict_one

WS = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Palmprint Recognition · Tongji", layout="wide")
st.title("Palmprint Recognition | Tongji")
st.caption(
    "Nghiên cứu 1:N closed-set · Official ROI · Không phải hệ thống xác thực danh tính thực tế"
)
st.info(
    "Ảnh ROI trong Tongji do tác giả dataset cung cấp. Không trích ROI từ webcam/ảnh toàn bàn tay."
)

available = [m for m in ("knn", "svm", "crc") if (WS / "artifacts" / f"{m}.joblib").exists()]
available.extend(
    f"resnet18_seed{p.name.split('seed')[1].split('.')[0]}"
    for p in sorted((WS / "artifacts").glob("resnet18_seed*.pth"))
)
for p in sorted((WS / "artifacts").glob("resnet18_seed*_gallery512.npz")):
    seed = p.stem.removeprefix("resnet18_seed").removesuffix("_gallery512")
    if seed.isdigit():
        available.extend(
            [f"resnet18_seed{seed}_embedding_nearest", f"resnet18_seed{seed}_embedding_centroid"]
        )
if not available:
    st.warning("Chưa có model artifact. Chạy prepare → cv → train trước khi mở demo.")
    st.stop()
model_name = st.selectbox("Thuật toán", available)
selection = st.radio(
    "Nguồn truy vấn", ["Chọn Official ROI từ dataset", "Tải ảnh ROI hợp lệ"], horizontal=True
)
if selection == "Chọn Official ROI từ dataset":
    try:
        df, info = load_manifest(WS)
        test = df[df.session == "session2"].copy()
        uid = st.selectbox("Ảnh session2", test.sample_uid.tolist())
        rec = test.loc[test.sample_uid == uid].iloc[0]
        path = resolve_path(info["data_root"], rec.path)
        true_id = int(rec.palm_id)
    except (FileNotFoundError, RuntimeError) as e:
        st.error(str(e))
        st.stop()
else:
    data = st.file_uploader(
        "Official ROI (PNG, BMP, JPEG, TIFF)", type=["png", "bmp", "jpg", "jpeg", "tif", "tiff"]
    )
    if data is None:
        st.stop()
    if len(data.getvalue()) > 10 * 1024 * 1024:
        st.error("File vượt giới hạn 10 MB")
        st.stop()
    suffix = "." + data.name.rsplit(".", 1)[-1].lower()
    image_bytes = data.getvalue()
    path = None
    true_id = None
left, right = st.columns([1, 2])
with left:
    try:
        st.image(
            Image.open(path) if path is not None else Image.open(BytesIO(image_bytes)),
            caption="Official ROI đầu vào",
            width=280,
        )
    except Exception as exc:
        st.error(f"Ảnh không hợp lệ: {exc}")
        st.stop()
if st.button("Nhận dạng", type="primary"):
    try:

        def run_prediction(img_path):
            if "_embedding_" in model_name:
                from palmprint.evaluation.embedding import predict_one_embedding

                head, strategy = model_name.split("_embedding_", 1)
                seed = int(head.removeprefix("resnet18_seed"))
                return predict_one_embedding(WS, seed, img_path, strategy=strategy)
            if model_name.startswith("resnet18_seed"):
                from palmprint.evaluation.cnn import predict_one_cnn

                return predict_one_cnn(WS, int(model_name.removeprefix("resnet18_seed")), img_path)
            return predict_one(WS, model_name, img_path)

        if path is None:
            with tempfile.TemporaryDirectory(prefix="palmprint_upload_") as td:
                local_file = Path(td) / ("uploaded" + suffix)
                local_file.write_bytes(image_bytes)
                prediction = run_prediction(local_file)
        else:
            prediction = run_prediction(path)
    except Exception as exc:
        st.error(f"Dự đoán thất bại: {exc}")
        st.stop()
    with right:
        st.metric("Palm ID dự đoán", prediction["predicted_palm_id"])
        st.caption(f"Thời gian từ ảnh đến dự đoán: {prediction['elapsed_ms']:.1f} ms (máy demo)")
        if true_id is not None:
            st.write(
                f"Nhãn thực: **{true_id}** · "
                + ("Dự đoán đúng" if true_id == prediction["predicted_palm_id"] else "Dự đoán sai")
            )
        out_frame = pd.DataFrame(
            {
                "Thứ hạng": list(range(1, len(prediction["top5"]) + 1)),
                "Palm ID": prediction["top5"],
                "Điểm thô (thấp tốt hơn)": prediction["scores_lower_is_better"],
            }
        )
        st.dataframe(out_frame, hide_index=True)
        st.caption("Điểm khoảng cách/negative margin/residual KHÔNG phải xác suất %.")
        if selection.startswith("Chọn"):
            st.subheader("Ảnh gallery đối chiếu (session1)")
            cols = st.columns(min(5, len(prediction["top5"])))
            for col, palm in zip(cols, prediction["top5"]):
                candidate = df[(df.session == "session1") & (df.palm_id == palm)].iloc[0]
                col.image(
                    Image.open(resolve_path(info["data_root"], candidate.path)),
                    caption=f"Palm {palm}",
                    use_container_width=True,
                )
if selection != "Chọn Official ROI từ dataset":
    st.caption(
        "Ảnh tải lên chỉ tạo file tạm trong lúc dự đoán và được dọn sau khi xử lý; không được thêm vào dataset."
    )

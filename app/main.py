"""Streamlit Web Demo: Tongji Contactless Palmprint Recognition.

BÀI TẬP LỚN CUỐI KỲ — MÔN NHẬN DẠNG MẪU (Đề tài 09 · Nhóm 01)
Chạy: streamlit run app/main.py
"""

from __future__ import annotations

import random
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from palmprint.data.dataset import resolve_path
from palmprint.data.preparation import load_manifest
from palmprint.features.compcode import CompCodeConfig, CompCodeExtractor
from palmprint.training.classical import predict_one

WS = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hệ thống Nhận diện Lòng bàn tay Tongji | Nhóm 01",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .success-box {
        background-color: #ECFDF5;
        border: 1.5px solid #10B981;
        border-radius: 8px;
        padding: 14px;
        color: #065F46;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FEF2F2;
        border: 1.5px solid #EF4444;
        border-radius: 8px;
        padding: 14px;
        color: #991B1B;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_manifest():
    return load_manifest(WS)


@st.cache_resource
def get_compcode_extractor():
    return CompCodeExtractor(CompCodeConfig())


# ---------------------------------------------------------
# Sidebar: Setup & Model Selection
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hand.png", width=64)
    st.markdown("### **Tongji Palmprint ID**")
    st.caption("BÀI TẬP LỚN CUỐI KỲ · MÔN NHẬN DẠNG MẪU\n**Đề tài 09 — Nhóm 01**")
    st.markdown("---")

    # Discover and build friendly names for artifacts
    MODEL_MAP = {
        "crc": "M3 — CRC_RLS (Chính xác nhất · 98.27%)",
        "svm": "M2 — Linear SVM (Nhanh nhất · 3.71 ms · 97.37%)",
        "knn": "M1 — 1-NN Baseline (K-láng giềng · 94.43%)",
        "resnet18_seed17_embedding_nearest": "M4-E2 — ResNet18 Nearest Cosine (512-D · 73.74%)",
        "resnet18_seed17_embedding_centroid": "M4-E2 — ResNet18 Centroid Cosine (512-D · 68.83%)",
        "resnet18_seed17": "M4-E1 — ResNet18 Logits (Softmax 600 lớp · 64.36%)",
    }

    available_keys = []
    for k in ("crc", "svm", "knn"):
        if (WS / "artifacts" / f"{k}.joblib").exists():
            available_keys.append(k)
    for p in sorted((WS / "artifacts").glob("resnet18_seed*_gallery512.npz")):
        seed = p.stem.removeprefix("resnet18_seed").removesuffix("_gallery512")
        if seed == "17":
            available_keys.extend([
                f"resnet18_seed{seed}_embedding_nearest",
                f"resnet18_seed{seed}_embedding_centroid",
            ])
    if (WS / "artifacts/resnet18_seed17.pth").exists():
        available_keys.append("resnet18_seed17")

    selected_display = st.selectbox(
        "🧠 Thuật toán nhận dạng:",
        options=[MODEL_MAP.get(k, k) for k in available_keys],
        index=0,
    )
    # Reverse lookup key
    selected_model_key = [k for k in available_keys if MODEL_MAP.get(k, k) == selected_display][0]

    st.markdown("---")
    st.markdown("#### ⚙️ Thông số hệ thống:")
    st.markdown(
        """
        - **Đối tượng:** 600 Palm ID
        - **Dữ liệu:** Tongji Contactless ROI
        - **Giao thức:** Closed-set 1:N
        - **Chống rò rỉ:** Khóa mô hình tuyệt đối
        """
    )
    st.markdown("---")
    st.caption("Repository: `hloc1810/tongji-palmprint-recognition`")


# ---------------------------------------------------------
# Main Interface Tabs
# ---------------------------------------------------------
st.markdown('<div class="main-header">🖐️ Hệ thống Nhận diện Lòng bàn tay không tiếp xúc</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Contactless Palmprint Identification trên bộ dữ liệu chuẩn Tongji University (600 Palm ID · Closed-set 1:N)</div>', unsafe_allow_html=True)

tab_demo, tab_feature, tab_benchmark, tab_about = st.tabs([
    "🔍 Nhận dạng Sinh trắc",
    "🔬 Trực quan hóa Đặc trưng",
    "📊 Bảng vàng Benchmark",
    "📖 Thông tin Đề tài",
])


# =========================================================
# TAB 1: NHẬN DẠNG SINH TRẮC
# =========================================================
with tab_demo:
    df_manifest, dataset_info = get_cached_manifest()
    test_samples = df_manifest[df_manifest.session == "session2"].copy()

    col_input, col_result = st.columns([1, 1.3], gap="large")

    with col_input:
        st.subheader("1. Chọn mẫu truy vấn (Query Probe)")
        source_mode = st.radio(
            "Nguồn ảnh:",
            ["Từ tập kiểm thử Session 2 (6.000 ảnh)", "Tải ảnh ROI lên (Upload)"],
            horizontal=True,
        )

        query_image_path = None
        query_image_bytes = None
        true_palm_id = None
        sample_name = ""

        if source_mode.startswith("Từ tập kiểm thử"):
            # Random pick button for quick demo during defense
            c_select, c_rand = st.columns([2.5, 1.2])
            with c_rand:
                st.write("")
                st.write("")
                if st.button("🎲 Ngẫu nhiên"):
                    st.session_state["selected_probe_uid"] = random.choice(test_samples.sample_uid.tolist())

            default_uid = st.session_state.get("selected_probe_uid", test_samples.sample_uid.iloc[0])
            probe_uids = test_samples.sample_uid.tolist()
            default_index = probe_uids.index(default_uid) if default_uid in probe_uids else 0

            with c_select:
                selected_uid = st.selectbox(
                    "Danh sách ảnh Session 2:",
                    probe_uids,
                    index=default_index,
                )

            rec = test_samples.loc[test_samples.sample_uid == selected_uid].iloc[0]
            query_image_path = resolve_path(dataset_info["data_root"], rec.path)
            true_palm_id = int(rec.palm_id)
            sample_name = selected_uid

            img_preview = Image.open(query_image_path)
            st.image(img_preview, caption=f"Mẫu truy vấn: {sample_name} (Ground Truth: Palm ID {true_palm_id})", width=220)

        else:
            uploaded = st.file_uploader(
                "Chọn file ảnh ROI 128×128 (BMP, PNG, JPG):",
                type=["bmp", "png", "jpg", "jpeg"],
            )
            if uploaded:
                query_image_bytes = uploaded.getvalue()
                suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
                img_preview = Image.open(BytesIO(query_image_bytes))
                sample_name = uploaded.name
                st.image(img_preview, caption=f"Ảnh tải lên: {sample_name}", width=220)

        btn_run = st.button("🚀 BẮT ĐẦU NHẬN DẠNG", type="primary", use_container_width=True)

    with col_result:
        st.subheader("2. Kết quả Đối soát & Nhận diện")

        if btn_run:
            if query_image_path is None and query_image_bytes is None:
                st.warning("Vui lòng chọn hoặc tải ảnh lên trước khi nhận dạng.")
            else:
                with st.spinner("Đang trích xuất đặc trưng và so khớp..."):
                    try:
                        def execute_predict(img_p):
                            if "_embedding_" in selected_model_key:
                                from palmprint.evaluation.embedding import predict_one_embedding
                                head, strategy = selected_model_key.split("_embedding_", 1)
                                seed = int(head.removeprefix("resnet18_seed"))
                                return predict_one_embedding(WS, seed, img_p, strategy=strategy)
                            if selected_model_key.startswith("resnet18_seed"):
                                from palmprint.evaluation.cnn import predict_one_cnn
                                return predict_one_cnn(WS, int(selected_model_key.removeprefix("resnet18_seed")), img_p)
                            return predict_one(WS, selected_model_key, img_p)

                        if query_image_path is not None:
                            pred = execute_predict(query_image_path)
                        else:
                            with tempfile.TemporaryDirectory(prefix="palm_demo_") as td:
                                tfile = Path(td) / ("query" + suffix)
                                tfile.write_bytes(query_image_bytes)
                                pred = execute_predict(tfile)

                        pred_id = pred["predicted_palm_id"]
                        elapsed = pred["elapsed_ms"]
                        top5_ids = pred["top5"]
                        top5_scores = pred["scores_lower_is_better"]

                        # Status Banner
                        if true_palm_id is not None:
                            if pred_id == true_palm_id:
                                st.markdown(
                                    f'<div class="success-box">✅ XÁC NHẬN CHÍNH XÁC: Nhận diện thành công Palm ID #{pred_id} (Trùng khớp 100% với nhãn thật #{true_palm_id})</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f'<div class="error-box">⚠️ DỰ ĐOÁN SAI: Hệ thống chọn Palm ID #{pred_id} (Nhãn thực tế là Palm ID #{true_palm_id})</div>',
                                    unsafe_allow_html=True,
                                )

                        # Metrics row
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Palm ID Dự đoán", f"#{pred_id}")
                        m2.metric("Thời gian suy luận", f"{elapsed:.2f} ms")
                        m3.metric("Thuật toán", selected_model_key.split("_")[0].upper())

                        # Top-5 Table
                        st.markdown("##### 🏆 Bảng xếp hạng Top-5 ứng viên gần nhất:")
                        score_label = "Khoảng cách / Sai số (Càng nhỏ càng khớp)"
                        df_top5 = pd.DataFrame({
                            "Thứ hạng": [f"Top {i+1}" for i in range(len(top5_ids))],
                            "Palm ID": [f"Palm #{pid}" for pid in top5_ids],
                            score_label: [f"{s:.4f}" for s in top5_scores],
                            "Trạng thái": ["✅ Khớp nhất (Top-1)" if i == 0 else "—" for i in range(len(top5_ids))],
                        })
                        st.dataframe(df_top5, use_container_width=True, hide_index=True)

                        # Gallery Matching Side-by-Side Verification
                        st.markdown("##### 🖼️ Mẫu đăng ký đối chiếu trong Gallery (Session 1):")
                        cols_gal = st.columns(min(5, len(top5_ids)))
                        for idx, (col_g, pid) in enumerate(zip(cols_gal, top5_ids)):
                            match_cand = df_manifest[(df_manifest.session == "session1") & (df_manifest.palm_id == pid)].iloc[0]
                            cand_path = resolve_path(dataset_info["data_root"], match_cand.path)
                            with col_g:
                                is_top1 = (idx == 0)
                                st.image(
                                    Image.open(cand_path),
                                    caption=f"{'👉 ' if is_top1 else ''}Palm #{pid} (Top {idx+1})",
                                    use_container_width=True,
                                )

                    except Exception as e:
                        st.error(f"Lỗi khi thực hiện nhận dạng: {e}")
        else:
            st.info("👈 Hãy chọn một ảnh bên trái rồi bấm **BẮT ĐẦU NHẬN DẠNG** để đối soát thời gian thực.")


# =========================================================
# TAB 2: TRỰC QUAN HÓA ĐẶC TRƯNG COMPCODE
# =========================================================
with tab_feature:
    st.subheader("🔬 Cơ chế Hoạt động của Trích xuất Đặc trưng Competitive Code (CompCode 600-D)")
    st.write(
        "Khám phá 'bên trong mô hình': Xem cách bộ lọc Gabor 6 hướng phản ứng với vân tay, tạo bản đồ hướng và sinh vector đặc trưng 600 chiều."
    )

    extractor = get_compcode_extractor()

    # Load image for visualization
    sample_for_vis = query_image_path if query_image_path is not None else (WS / "data/raw/session1/00001.bmp")
    gray_roi = cv2.imread(str(sample_for_vis), cv2.IMREAD_GRAYSCALE)

    c_v1, c_v2, c_v3 = st.columns(3)
    with c_v1:
        st.markdown("**1. ROI Đầu vào (128×128)**")
        st.image(gray_roi, caption="Official ROI Grayscale", width=180)

    with c_v2:
        st.markdown("**2. Bản đồ Competitive Code**")
        ori_map = extractor.orientation_map(gray_roi)
        fig_ori, ax_ori = plt.subplots(figsize=(3, 3), dpi=150)
        im = ax_ori.imshow(ori_map, cmap="tab10", vmin=0, vmax=5)
        ax_ori.axis("off")
        ax_ori.set_title("Orientation Map (0–5)", fontsize=8)
        st.pyplot(fig_ori)
        plt.close(fig_ori)

    with c_v3:
        st.markdown("**3. Phân hoạch Lưới 10×10 khối**")
        grid_img = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2RGB)
        for r in range(0, 128, 14):
            cv2.line(grid_img, (0, r), (128, r), (255, 60, 60), 1)
        for c in range(0, 128, 14):
            cv2.line(grid_img, (c, 0), (c, 128), (255, 60, 60), 1)
        st.image(grid_img, caption="100 khối (14×14 px/khối)", width=180)

    st.markdown("**4. Vector Đặc trưng Toàn cục 600 Chiều (100 khối × 6 hướng):**")
    vec_600 = extractor.transform(gray_roi)
    fig_vec, ax_vec = plt.subplots(figsize=(10, 2.5), dpi=150)
    ax_vec.plot(vec_600, color="#1E3A8A", linewidth=0.9)
    ax_vec.fill_between(range(len(vec_600)), vec_600, color="#3B82F6", alpha=0.3)
    ax_vec.set_xlim(0, 600)
    ax_vec.set_xlabel("Chỉ số chiều đặc trưng (0 đến 599)", fontsize=8)
    ax_vec.set_ylabel("Giá trị chuẩn hóa L2", fontsize=8)
    ax_vec.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_vec)
    plt.close(fig_vec)


# =========================================================
# TAB 3: BẢNG VÀNG BENCHMARK & ĐỒ THỊ
# =========================================================
with tab_benchmark:
    st.subheader("📊 Bảng vàng Thành tích Thực nghiệm (Benchmark trên 6.000 probe Session 2)")

    # Key metric cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Độ chính xác cao nhất", "98.27%", "CRC_RLS (M3)")
    k2.metric("Tốc độ nhanh nhất", "3.71 ms", "Linear SVM (M2)")
    k3.metric("Số lớp nhận dạng", "600 Palm ID", "Closed-set 1:N")
    k4.metric("Dữ liệu thử nghiệm", "6.000 ảnh", "Session 2 Probe")

    st.markdown("---")
    st.markdown("##### 📋 Bảng đối chiếu toàn bộ các Pipeline:")

    table_path = WS / "reports/tables/tongji_benchmark_summary.csv"
    if table_path.exists():
        df_bench = pd.read_csv(table_path)
        st.dataframe(df_bench, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📈 Các đồ thị phân tích chuyên sâu (Xuất bản 300 DPI):")

    c_fig1, c_fig2 = st.columns(2)
    with c_fig1:
        tradeoff_img = WS / "reports/figures/accuracy_vs_latency_tradeoff.png"
        if tradeoff_img.exists():
            st.image(str(tradeoff_img), caption="Đồ thị Đánh đổi Độ chính xác vs Độ trễ CPU (Pareto Frontier)", use_container_width=True)

    with c_fig2:
        model_cmp_img = WS / "reports/figures/model_comparison.png"
        if model_cmp_img.exists():
            st.image(str(model_cmp_img), caption="So sánh Rank-1, Rank-5 và Macro-F1 giữa các mô hình", use_container_width=True)

    c_fig3, c_fig4 = st.columns(2)
    with c_fig3:
        cmc_img = WS / "reports/figures/cnn_vs_classical_cmc_rank20.png"
        if cmc_img.exists():
            st.image(str(cmc_img), caption="Đường cong CMC Rank 1–20 giữa Cổ điển và Deep Learning", use_container_width=True)

    with c_fig4:
        error_img = WS / "reports/figures/qualitative_predictions_success_failure.png"
        if error_img.exists():
            st.image(str(error_img), caption="Trực quan hóa các ca nhận dạng đúng và sai thực tế", use_container_width=True)


# =========================================================
# TAB 4: THÔNG TIN ĐỀ TÀI & NHÓM
# =========================================================
with tab_about:
    st.subheader("📖 BÀI TẬP LỚN CUỐI KỲ — MÔN NHẬN DẠNG MẪU (PATTERN RECOGNITION)")
    st.markdown(
        """
        - **Đề tài số 09:** Nhận diện lòng bàn tay không tiếp xúc (*Contactless Palmprint Recognition*)
        - **Nhóm thực hiện:** Nhóm 01
        - **Repository:** [`github.com/hloc1810/tongji-palmprint-recognition`](https://github.com/hloc1810/tongji-palmprint-recognition)

        ---
        #### 🎯 Tóm tắt đóng góp chính của đề tài:
        1. **Quy trình chuẩn mực chống rò rỉ dữ liệu (No Data Leakage):**
           - Áp dụng giao thức nghiêm ngặt **Cross-Session (Session 1 train, Session 2 test)**.
           - Chọn tham số chỉ bằng **5-fold CV nội bộ trên Session 1**. Khóa toàn bộ mô hình trước khi đánh giá trên Session 2.
        2. **Cơ chế đặc trưng Competitive Code (CompCode 600-D):**
           - Dùng 6 bộ lọc Gabor với cấu hình `zero_dc = True` tự động triệt tiêu sai lệch ánh sáng nền mà không làm mờ nếp vân.
        3. **Tìm ra 2 mô hình tối ưu tuyệt đối:**
           - **CRC_RLS (M3)** đạt độ chính xác cao nhất (**98.27%** Rank-1) nhờ cơ chế biểu diễn cộng tác.
           - **Linear SVM (M2)** đạt tốc độ suy luận nhanh nhất (**3.71 ms/ảnh**, model 2.7 MB), tối ưu cho triển khai phần cứng nhúng.
        4. **Khám phá học thuật về Deep Metric Learning:**
           - Chứng minh được không gian **Cosine Embedding (Nearest)** giúp ResNet18 tăng vọt **+9.38% Rank-1** so với tầng phân loại Softmax bị overfitting khi dữ liệu ít mẫu.
        """
    )

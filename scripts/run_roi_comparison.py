"""Paired 10-palm ROI pilot: existing CompCode + existing Euclidean 1-NN.

Run from the repository root. Never writes to main benchmark artifacts/metadata.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palmprint.data.dataset import EXTENSIONS, read_gray
from palmprint.data.tongji_roi import ROIConfig, ROIExtractionError, draw_overlay, extract_roi
from palmprint.features.compcode import CompCodeConfig, CompCodeExtractor
from palmprint.models.classical import NearestTemplate
from palmprint.utils import hash_file, read_yaml

WS = Path(__file__).resolve().parents[1]


def image_index(root: Path, session: str) -> dict[str, Path]:
    folder = root / session
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    index = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in index:
                raise ValueError(f"Duplicate stem in {folder}: {path.stem}")
            index[path.stem] = path.resolve()
    return index


def select_samples(original: Path, official: Path, palms: list[int]) -> pd.DataFrame:
    """Require all 20 images for every preselected ID; never replace failed crops."""
    rows = []
    for session in ("session1", "session2"):
        originals, officials = image_index(original, session), image_index(official, session)
        for palm in palms:
            for sample in range(1, 11):
                stem = f"{(palm - 1) * 10 + sample:05d}"
                if stem not in originals or stem not in officials:
                    raise ValueError(f"Unpaired/missing sample: {session}/{stem}")
                rows.append(
                    {
                        "sample_uid": f"{session}/{stem}",
                        "session": session,
                        "palm_id": palm,
                        "sample": sample,
                        "stem": stem,
                        "original_path": str(originals[stem]),
                        "official_path": str(officials[stem]),
                        "original_sha256": hash_file(originals[stem]),
                        "official_sha256": hash_file(officials[stem]),
                    }
                )
    return pd.DataFrame(rows)


def load_original(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def evaluate(features: dict, samples: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    gallery = samples[(samples.session == "session1") & samples.sample_uid.isin(features)]
    probes = samples[samples.session == "session2"]
    predictions = probes[["sample_uid", "palm_id"]].copy()
    predictions["predicted_palm_id"] = -1
    predictions["roi_available"] = predictions.sample_uid.isin(features)
    valid = predictions[predictions.roi_available]
    elapsed = 0.0
    if len(gallery) and len(valid):
        model = NearestTemplate().fit(
            np.stack([features[uid] for uid in gallery.sample_uid]), gallery.palm_id.to_numpy()
        )
        query = np.stack([features[uid] for uid in valid.sample_uid])
        t0 = perf_counter()
        scores = model.scores(query)
        elapsed = (perf_counter() - t0) * 1000
        predictions.loc[valid.index, "predicted_palm_id"] = model.classes_[
            np.argmin(scores, axis=1)
        ]
    predictions["correct"] = predictions.palm_id == predictions.predicted_palm_id
    return {
        "rank1": float(predictions.correct.mean()),
        "correct": int(predictions.correct.sum()),
        "probe_denominator": len(probes),
        "valid_probes": len(valid),
        "gallery_images": len(gallery),
        "gallery_classes": int(gallery.palm_id.nunique()),
        "matching_batch_ms": elapsed,
        "matching_ms_per_valid_probe": elapsed / len(valid) if len(valid) else None,
    }, predictions


def save_audit_sheets(samples, audits, images, officials, customs, out):
    """Every selected image is visible; no cherry-picked-only visual evidence."""
    audit_by_uid = {row["sample_uid"]: row for row in audits}
    for session in ("session1", "session2"):
        part = samples[samples.session == session].reset_index(drop=True)
        for start in (0, 50):
            tiles = []
            for row in part.iloc[start : start + 50].itertuples():
                audit = audit_by_uid[row.sample_uid]
                original = images[row.sample_uid]
                if audit["success"]:
                    original = draw_overlay(original, audit["landmarks"])
                tile = np.full((222, 200, 3), 245, np.uint8)
                tile[:150] = cv2.resize(original, (200, 150))
                tile[153:217, 4:68] = cv2.cvtColor(
                    cv2.resize(officials[row.sample_uid], (64, 64)), cv2.COLOR_GRAY2BGR
                )
                if row.sample_uid in customs:
                    tile[153:217, 72:136] = cv2.cvtColor(
                        cv2.resize(customs[row.sample_uid], (64, 64)), cv2.COLOR_GRAY2BGR
                    )
                color = (0, 80, 0) if audit["success"] else (0, 0, 220)
                cv2.putText(tile, row.stem, (140, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
                cv2.putText(
                    tile,
                    "OK" if audit["success"] else "FAIL",
                    (140, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
                tiles.append(tile)
            canvas = np.concatenate(
                [np.concatenate(tiles[i : i + 10], axis=1) for i in range(0, len(tiles), 10)],
                axis=0,
            )
            cv2.imwrite(str(out / f"audit_{session}_{start // 50 + 1}.jpg"), canvas)


def save_examples(samples, audits, images, officials, customs, predictions, out):
    by_uid = {a["sample_uid"]: a for a in audits}
    accepted = sorted((a for a in audits if a["success"]), key=lambda a: a["ssim_official"])
    rejected = [a for a in audits if not a["success"]]
    errors = predictions[~predictions.correct].sample_uid.tolist()
    # Include reference similarity extremes, failures, and recognition errors separately.
    chosen = []
    if accepted:
        chosen += [
            (accepted[-1]["sample_uid"], "High reference similarity"),
            (accepted[0]["sample_uid"], "Low similarity; inspect geometry"),
        ]
    chosen += [(a["sample_uid"], "Rejected crop: " + a["reason"]) for a in rejected[:2]]
    chosen += [(uid, "Recognition error; crop may still be valid") for uid in errors[:2]]
    chosen = list({uid: title for uid, title in chosen}.items())
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 2.9 * len(chosen)), squeeze=False)
    for i, (uid, title) in enumerate(chosen):
        a = by_uid[uid]
        original = draw_overlay(images[uid], a["landmarks"]) if a["success"] else images[uid]
        axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(uid + "\n" + textwrap.fill(title, width=42), fontsize=8)
        axes[i, 1].imshow(officials[uid], cmap="gray", vmin=0, vmax=255)
        axes[i, 1].set_title("Official ROI")
        if uid in customs:
            axes[i, 2].imshow(customs[uid], cmap="gray", vmin=0, vmax=255)
            axes[i, 2].set_title(f"Custom ROI | SSIM={a['ssim_official']:.3f}")
        else:
            axes[i, 2].text(0.5, 0.5, a["reason"], ha="center", wrap=True)
            axes[i, 2].set_title("No substitute ROI")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "roi_examples.png", dpi=160)
    plt.close(fig)
    (out / "example_selection.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")


def write_report(summary: dict, out: Path):
    a, b = summary["official"], summary["custom"]
    mean, median, p95 = (b[k] for k in ("roi_mean_ms", "roi_median_ms", "roi_p95_ms"))
    report = f"""# Thử nghiệm ROI Tongji 2017 trên 10 Palm ID

## Phạm vi và kết quả

Thực nghiệm thăm dò closed-set 1:N, 10 Palm ID, 200 ảnh ghép cặp.
Session 1: 100 gallery; Session 2: 100 probe. ID được chọn ngẫu nhiên không hoàn lại
với NumPy default_rng(seed={summary["seed"]}), trước khi đánh giá, không chọn lại theo kết quả:
**{", ".join(map(str, summary["palm_ids"]))}**.

| Chỉ số | Official ROI | Custom ROI |
|---|---:|---:|
| Số Palm ID đã chọn | 10 | 10 |
| Số ảnh đầu vào | 200 | 200 |
| Gallery dùng được | {a["gallery_images"]} | {b["gallery_images"]} |
| Palm ID có template gallery | {a["gallery_classes"]} | {b["gallery_classes"]} |
| Probe cắt được | 100 | {b["valid_probes"]} |
| Tỷ lệ cắt ROI thành công theo kiểm tra tự động | Không áp dụng | {b["roi_accepted"]}/200 ({b["roi_accepted"] / 2:.2f}%) |
| Rank-1, mẫu số cố định 100 probe | {a["correct"]}/100 ({a["rank1"] * 100:.2f}%) | {b["correct"]}/100 ({b["rank1"] * 100:.2f}%) |
| Thời gian xử lý ROI trung bình/ảnh | Không áp dụng | {mean:.2f} ms |
| ROI median / p95 | Không áp dụng | {median:.2f} / {p95:.2f} ms |
| Tổng thời gian thuật toán ROI, 200 lần thử | Không áp dụng | {b["roi_total_ms"] / 1000:.3f} s |

Không so trực tiếp với **94,43% trên benchmark 600 lớp**. Cả hai Rank-1 trong bảng
đều được chạy lại trên đúng bộ 10 lớp này. Đây là pilot, không suy rộng thành kết quả 600 lớp.

"Thành công tự động" nghĩa là tìm được bộ kẽ ngón hợp lệ, tiếp tuyến, hai giao biên,
hình vuông nằm trong ảnh và ít nhất 98% ROI thuộc hand mask. Đây là tỷ lệ thuật toán
chấp nhận crop, **không phải tỷ lệ đúng giải phẫu đã gán nhãn thủ công**. SSIM với Official ROI
chỉ hỗ trợ xem ảnh, không dùng chọn mẫu, chỉnh tham số hoặc quyết định crop thành công.

Probe thất bại được tính sai trong mẫu số 100; gallery thất bại không được dùng.
Không thay bằng Official ROI, không crop tâm dự phòng, không loại probe khó khỏi mẫu số.

## Thuật toán và đối chiếu paper

Nguồn: Zhang et al., *Towards contactless palmprint recognition*, Pattern Recognition 69
(2017), 199–212, DOI 10.1016/j.patcog.2017.04.016, Section 4, Fig. 5, Table 2, trang 204–205.
[PDF tác giả](https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf).

1. Chuyển ảnh gốc sang grayscale OpenCV; Gaussian 5×5, sigma=1.5 để phát hiện mask.
   Ngưỡng cố định 25/255, giữ contour ngoài lớn nhất và lấp lỗ bên trong.
2. Tính góc lõm trên contour ở hai phía cách 0.0375×chiều rộng ảnh theo số điểm contour.
   Chọn ba kẽ phía trên có độ lõm lớn, khoảng cách tương đối đều và gần thẳng hàng;
   lấy kẽ ngoài cùng (trỏ–giữa và áp út–út). Tính tiếp tuyến đỡ chung phía dưới
   hai đoạn biên cục bộ bằng nghiệm số; **X1, X2 là tiếp điểm**, không chỉ là đỉnh lõm.
3. O=(X1+X2)/2; u=(X2−X1)/d; v vuông góc hướng xuống lòng bàn tay; d=|X1X2|.
   Cắt contour bằng đường O+0.60d·v+t·u để lấy O1, O2; w=|O1O2|.
4. Cạnh s=0.65w. Tâm cạnh trên T=O+0.10w·v. Bốn góc là
   T−s/2·u, T+s/2·u, T+s/2·u+s·v, T−s/2·u+s·v.
   Warp affine bilinear **ảnh grayscale gốc**, về 128×128, không blur texture đầu ra.

**Khác biệt ngay trong paper:** Table 2 bước 7 in khoảng dịch `c2|O1O2|`, trong khi
đoạn mô tả Section 4 trang 205 ghi `c3|O1O2|` và c3=0.10. Cài đặt này theo đoạn mô tả
và Hình 5: khoảng dịch đến **cạnh gần trục X nhất**, không phải tâm hình vuông.

Paper không nêu đầy đủ ngưỡng segmentation, kernel Gaussian, cách tìm kẽ và nghiệm tiếp tuyến.
Các lựa chọn kỹ thuật trên là phần tự triển khai, đã cố định khi xem Session 1 trước khi
chạy Session 2. **Không tuyên bố tái lập từng pixel hay tương đương mã nguồn tác giả.**
Giả định ảnh đúng tư thế chụp Tongji (ngón hướng lên, nền tối); không phải bộ phát hiện
bàn tay tự do ở mọi góc quay. Các phép chuẩn hóa vị trí/góc/tỷ lệ diễn ra sau phát hiện kẽ.

## CompCode và 1-NN được giữ nguyên

Dùng `CompCodeExtractor` với `configs/compcode.yaml` hiện có: 6 Gabor, 600-D spatial
histogram, chuẩn hóa L1 theo block rồi L2 toàn vector. Dùng chính `NearestTemplate` hiện có,
score lớp bằng khoảng cách Euclidean bình phương nhỏ nhất đến template của lớp,
lấy argmin trên các Palm ID tăng dần khi hòa. Không PCA, LDA, fit scaler hay mô hình mới.
CompCode của dự án vốn là bản paper-inspired, chưa xác minh parity MATLAB.

Thời gian ROI đo bằng perf_counter trên ảnh đã decode, gồm segmentation đến warp và
kiểm tra hình học, gồm cả lần bị từ chối; không gồm đọc/ghi file, CompCode, 1-NN và vẽ ảnh.
Warm-up 3 ảnh Session 1, OpenCV một luồng. Mỗi ảnh đo một lần; đây là số đo trên máy này,
có thể dao động giữa các lần chạy. Chi tiết môi trường và thời gian các khâu nằm trong summary.json.

## Bằng chứng để kiểm tra

- [Ảnh minh họa](roi_examples.png): tương đồng cao/thấp với ROI chính thức;
  thêm crop bị từ chối hoặc nhận dạng sai nếu có. **Nhận dạng sai không đồng nghĩa crop sai.**
- [Kiểm tra Session 1, ID 1–5](audit_session1_1.jpg), [ID 6–10](audit_session1_2.jpg).
- [Kiểm tra Session 2, ID 1–5](audit_session2_1.jpg), [ID 6–10](audit_session2_2.jpg).
  Mỗi ô: ảnh gốc có điểm mốc; thumbnail dưới trái Official, dưới phải Custom.
  Vàng: tiếp tuyến X1X2; xanh lam: đường đo O1O2; xanh lá: hình vuông crop.
- `samples.csv`: 200 đường dẫn và SHA256; `protocol.json`: lựa chọn/thuật toán khóa trước đánh giá.
- `roi_audit.csv`, `landmarks.json`: trạng thái, lý do lỗi, thời gian, hình học từng ảnh.
- `predictions_official.csv`, `predictions_custom.csv`: đủ 100 probe mỗi nhánh.
- `comparison.csv`, `summary.json`: kết quả máy đọc được; `custom_roi/`: ảnh 128×128.

## Chạy lại

```powershell
.venv311/Scripts/python.exe scripts/run_roi_comparison.py --output outputs/roi_comparison_repeat
```

Có thể chỉ định `--original-root`, `--official-root`, `--output` và `--seed`.
Chọn thư mục output mới mỗi lần chạy; chương trình từ chối ghi đè một thực nghiệm đã lưu.
Nếu đổi seed/thuật toán sau khi xem kết quả, đó là thực nghiệm thăm dò mới; không dùng
Session 2 để chọn biến thể tốt nhất rồi trình bày như đánh giá độc lập.
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=WS / "data/Original Images")
    parser.add_argument("--official-root", type=Path, default=WS / "data/raw")
    parser.add_argument("--output", type=Path, default=WS / "outputs/roi_comparison")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Refuse accidental replacement of an existing experimental record.
    if (out / "summary.json").exists() or (out / "protocol.json").exists():
        raise FileExistsError(f"Existing experiment: {out}. Choose a fresh --output directory.")
    cv2.setNumThreads(1)
    palms = sorted(
        np.random.default_rng(args.seed).choice(np.arange(1, 601), 10, replace=False).tolist()
    )
    samples = select_samples(args.original_root, args.official_root, palms)
    samples.to_csv(out / "samples.csv", index=False)
    cfg = ROIConfig()
    feature_cfg = CompCodeConfig(**read_yaml(WS / "configs/compcode.yaml"))
    protocol = {
        "seed": args.seed,
        "palm_ids": palms,
        "roi_config": asdict(cfg),
        "compcode_config": asdict(feature_cfg),
        "classifier": "existing NearestTemplate Euclidean",
        "gallery": "session1",
        "probe": "session2",
        "probe_failure_policy": "count_incorrect",
        "sample_manifest_sha256": hash_file(out / "samples.csv"),
        "source_sha256": {
            str(p.relative_to(WS)): hash_file(p)
            for p in [
                Path(__file__).resolve(),
                WS / "src/palmprint/data/tongji_roi.py",
                WS / "src/palmprint/features/compcode.py",
                WS / "src/palmprint/models/classical.py",
            ]
        },
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for path in samples[samples.session == "session1"].original_path.iloc[:3]:
        try:
            extract_roi(load_original(path), cfg)
        except ROIExtractionError:
            pass
    extractor = CompCodeExtractor(feature_cfg)
    extractor.kernels  # Build filters outside feature timing.
    images, officials, customs = {}, {}, {}
    feats = {"official": {}, "custom": {}}
    feature_ms = {"official": [], "custom": []}
    audits = []
    for i, row in enumerate(samples.itertuples(index=False)):
        image = load_original(row.original_path)
        official = read_gray(row.official_path)
        images[row.sample_uid], officials[row.sample_uid] = image, official
        audit = {
            "sample_uid": row.sample_uid,
            "success": False,
            "reason": "",
            "landmarks": None,
            "ssim_official": None,
        }
        t0 = perf_counter()
        try:
            result = extract_roi(image, cfg)
            audit["roi_ms"] = (perf_counter() - t0) * 1000
            audit.update(success=True, landmarks=result.landmarks)
            customs[row.sample_uid] = result.roi
        except ROIExtractionError as error:
            audit["roi_ms"] = (perf_counter() - t0) * 1000
            audit["reason"] = str(error)
        if audit["success"]:
            folder = out / "custom_roi" / row.session
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"{row.stem}.png"), result.roi)
            audit["ssim_official"] = float(
                structural_similarity(official, result.roi, data_range=255)
            )
        for name, roi in (("official", official), ("custom", customs.get(row.sample_uid))):
            if roi is not None:
                t0 = perf_counter()
                feats[name][row.sample_uid] = extractor.transform(roi)
                feature_ms[name].append((perf_counter() - t0) * 1000)
        audits.append(audit)
        if (i + 1) % 50 == 0:
            print(f"ROI {i + 1}/200; accepted={sum(a['success'] for a in audits)}", flush=True)
    official_metrics, official_preds = evaluate(feats["official"], samples)
    custom_metrics, custom_preds = evaluate(feats["custom"], samples)
    times = np.array([a["roi_ms"] for a in audits])
    custom_metrics.update(
        roi_accepted=sum(a["success"] for a in audits),
        roi_total_ms=float(times.sum()),
        roi_mean_ms=float(times.mean()),
        roi_median_ms=float(np.median(times)),
        roi_p95_ms=float(np.percentile(times, 95)),
    )
    summary = {
        "seed": args.seed,
        "palm_ids": palms,
        "images": 200,
        "official": official_metrics,
        "custom": custom_metrics,
        "compcode_mean_ms": {k: float(np.mean(v)) if v else None for k, v in feature_ms.items()},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "opencv_threads": cv2.getNumThreads(),
        },
        "protocol_sha256": hash_file(out / "protocol.json"),
    }
    official_preds.to_csv(out / "predictions_official.csv", index=False)
    custom_preds.to_csv(out / "predictions_custom.csv", index=False)
    pd.DataFrame([{k: v for k, v in a.items() if k != "landmarks"} for a in audits]).to_csv(
        out / "roi_audit.csv", index=False
    )
    (out / "landmarks.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {"method": "Official ROI", "palm_ids": 10, "images": 200, **official_metrics},
            {"method": "Custom ROI", "palm_ids": 10, "images": 200, **custom_metrics},
        ]
    ).to_csv(out / "comparison.csv", index=False)
    save_audit_sheets(samples, audits, images, officials, customs, out)
    save_examples(samples, audits, images, officials, customs, custom_preds, out)
    write_report(summary, out)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

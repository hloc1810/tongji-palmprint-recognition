"""Create the Vietnamese phase-2 report and plots from completed experiment files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from palmprint.utils import atomic_json, hash_file, hash_object


def main():
    ws = Path.cwd()
    out = ws / "outputs/cnn_experiments"
    done = json.loads((out / "run_complete.json").read_text(encoding="utf-8"))
    protocol = json.loads((out / "protocol_before_cnn_test.json").read_text(encoding="utf-8"))
    result = json.loads((out / "recognition_results.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((out / "checkpoint_diagnostics.json").read_text(encoding="utf-8"))
    gap_audit = json.loads((out / "float32_gap_verification_audit.json").read_text(encoding="utf-8"))
    assert result["source_unchanged_since_registration"]
    current_hash = hash_object({str(p.relative_to(ws)): hash_file(p)
                               for p in sorted((ws / "src/palmprint").rglob("*.py"))})
    assert current_hash == protocol["source_sha256"]
    tables = ws / "reports/tables"
    figures = ws / "reports/figures"
    seeds = pd.read_csv(tables / "cnn_seed_results.csv")
    summary = pd.read_csv(tables / "cnn_seed_summary.csv")
    paired = pd.read_csv(tables / "cnn_mean_vs_classical_paired.csv")
    classical = pd.read_csv(tables / "tongji_benchmark.csv")
    comparison = pd.read_csv(tables / "model_comparison.csv")
    is_cnn = comparison.model.str.startswith("resnet18_seed")
    comparison["batch_timing_scope"] = np.where(is_cnn,
        "CNN batch: file IO + transforms + GPU forward", "classical batch: classifier scores only, CPU")
    comparison.loc[is_cnn, "note"] = (
        "CNN batch timing includes IO/preprocessing/GPU forward; not comparable with classical score-only batch timing. "
        "See cnn_cpu_latency.csv for single-image CPU timings.")
    comparison.to_csv(tables / "model_comparison.csv", index=False)
    latency = []
    for seed in protocol["seeds"]:
        m = json.loads((ws / f"outputs/runs/resnet18_seed{seed}/benchmark_cpu.json").read_text(encoding="utf-8"))
        latency.append({"model": f"resnet18_seed{seed}", "n_queries": m["n_queries"],
                        "median_ms": m["total_median_ms"], "p95_ms": m["total_p95_ms"]})
    pd.DataFrame(latency).to_csv(tables / "cnn_cpu_latency.csv", index=False)
    colors = ["#2468a0", "#ce6b25", "#3b8459"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex="col")
    for index, seed in enumerate(protocol["seeds"]):
        history = pd.read_csv(ws / f"outputs/runs/resnet18_seed{seed}/dev_history.csv")
        epochs = np.arange(1, len(history) + 1)
        axes[0, index].plot(epochs, history.train_loss, label="train CE", color=colors[0])
        axes[0, index].plot(epochs, history.dev_val_loss, label="session1 val CE", color=colors[1])
        axes[1, index].plot(epochs, history.train_rank1, label="train Rank-1", color=colors[0])
        axes[1, index].plot(epochs, history.dev_val_rank1, label="session1 val Rank-1", color=colors[1])
        boundary = int((history.stage == "head").sum()) + .5
        for row in range(2):
            axes[row, index].axvline(boundary, color="grey", ls="--", alpha=.7)
            axes[row, index].grid(alpha=.2)
            axes[row, index].legend(fontsize=8)
        axes[0, index].set_title(f"Seed {seed}: development only")
        axes[0, index].set_yscale("log")
        axes[1, index].set(xlabel="Development epoch (head then layer4)", ylim=(0, 1.03))
    axes[0, 0].set_ylabel("Cross-entropy (log scale)")
    axes[1, 0].set_ylabel("Rank-1 rate")
    fig.tight_layout()
    fig.savefig(figures / "cnn_training_curves.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, color in zip(("knn", "svm", "crc"), colors):
        curve = pd.read_csv(ws / f"outputs/runs/{name}/cmc_full.csv").head(20)
        ax.plot(curve.rank_k, curve.identification_rate, label=name, color=color)
    cnn_curves = []
    for seed in protocol["seeds"]:
        curve = pd.read_csv(ws / f"outputs/runs/resnet18_seed{seed}/cmc_full.csv").head(20)
        cnn_curves.append(curve.identification_rate.to_numpy())
        ax.plot(curve.rank_k, curve.identification_rate, color="#82479d", alpha=.3)
    ax.plot(np.arange(1, 21), np.mean(cnn_curves, axis=0), color="#82479d", lw=2.5,
            label="ResNet18: mean of 3 seed CMCs (no ensemble)")
    ax.set(xlabel="Rank k (distinct Palm IDs)", ylabel="Identification rate",
           title="Tongji session2: CMC ranks 1-20", xticks=[1, 2, 5, 10, 15, 20])
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "cnn_vs_classical_cmc_rank20.png", dpi=170)
    plt.close(fig)

    config = protocol["training"]
    lines = ["# Giai đoạn 2 — ResNet18: ba seed và so sánh với mô hình cổ điển", "",
             f"Hoàn tất: {done['completed_at']} (Asia/Saigon).", "",
             "Các số liệu dưới đây được đọc từ artifact và dự đoán của lần chạy thật trên Tongji. Không dùng kết quả synthetic hoặc lấy accuracy của tác giả làm kết quả nhóm.", "",
             "## Thiết kế đã chốt trước đánh giá CNN", "",
             "- 600 Palm ID, 6.000 ảnh session1 và 6.000 probe session2, cùng manifest với benchmark M1–M3.",
             "- ResNet18 pretrained ImageNet1K V1; ROI grayscale lặp ba kênh, resize trực tiếp 224×224, chuẩn hóa ImageNet; không dùng CompCode cho CNN.",
             f"- Seed cố định: {protocol['seeds']}; batch {config['batch_size']}; tối đa {config['head_epochs']} epoch head và {config['finetune_epochs']} epoch layer4; patience={config['patience']}.",
             "- Chọn stage/epoch trên fold0 session1: 4.800 train + 1.200 val. Ưu tiên val Rank-1, hòa thì CE thấp hơn; sau đó khởi tạo lại và refit toàn bộ 6.000 ảnh session1 theo lịch đã chọn.",
             f"- AdamW: head LR={config['head_lr']}; fine-tune layer4 LR={config['finetune_lr']}, fc LR={config['finetune_head_lr']}; weight decay={config['weight_decay']}; cosine LR, clip norm={config['grad_clip']}.",
             "- Giữ BatchNorm running statistics cố định. Train augmentation: rotation ±5°, brightness/contrast ±8%; validation/test không augmentation.",
             "- Đã khóa cả ba model CNN final trước khi đánh giá bất kỳ seed CNN nào trên session2. Không chọn seed tốt nhất bằng test; không ghép ensemble.",
             f"- Huấn luyện và đánh giá dùng {protocol['device']['name']} qua DirectML, float32. Torch CPU wheel + plugin DirectML vẫn thực thi convolution trên GPU; CUDA/AMP không được dùng.",
             "- Một phép toán AdamW `aten::lerp.Scalar_out` fallback về CPU. Checkpoint lưu tensor CPU; validation DirectML dùng no_grad để tránh lỗi inference tensors của backend.", "",
             "## Kết quả từng seed trên session2", "",
             "| Seed | Stage chọn | Epoch head / layer4 khi refit | Best val Rank-1 | Test Rank-1 | Test Rank-5 | Macro-F1 | Đúng / 6.000 | CI 95% Rank-1 |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in seeds.itertuples(index=False):
        lines.append(f"| {r.seed} | {r.selected_stage} | {r.head_epochs_final} / {r.finetune_epochs_final} | "
                     f"{100*r.dev_val_best_rank1:.4f}% | {100*r.rank1:.4f}% | {100*r.rank5:.4f}% | "
                     f"{100*r.macro_f1:.4f}% | {r.correct_probes} | [{100*r.rank1_ci95_lower:.4f}%; {100*r.rank1_ci95_upper:.4f}%] |")
    lines += ["", "CI từng run: bootstrap theo Palm ID, 1.000 lần, seed=42; số probe của mỗi run là 6.000.", "",
              "## Chẩn đoán khoảng cách lớn giữa session1 và session2", "",
              "Kết quả test thấp được giữ nguyên. Các kiểm tra dưới đây là chẩn đoán sau thực nghiệm, chỉ inference từ checkpoint đã khóa; không train lại, sửa tham số hoặc chọn seed bằng kết quả test.", "",
              "| Seed | Rank-1 đánh giá lại 6.000 ảnh session1, không augment | Số ROI đối chiếu CPU/GPU | Top-1 CPU/GPU khớp | Max sai khác logit |",
              "|---|---:|---:|---:|---:|"]
    for d in diagnostics:
        lines.append(f"| {d['seed']} | {100*d['session1_postfit_rank1_no_augmentation']:.4f}% | "
                     f"{d['cpu_gpu_parity_n']} | {100*d['top1_agreement_cpu_gpu']:.4f}% | {d['max_absolute_logit_difference']:.8f} |")
    lines += ["", "Tập đối chiếu CPU/GPU gồm ROI đầu tiên của mỗi Palm ID trong mỗi session: 600 ảnh/session, 1.200 ảnh/seed. Checkpoint nhận dạng tốt session1 và CPU/GPU nhất quán trên tập này; chưa thấy bằng chứng lỗi chuyển checkpoint hoặc backend gây ra mức giảm Rank-1 khoảng 35 điểm phần trăm. Kết quả cho thấy khả năng tổng quát hóa cross-session kém của cấu hình hiện tại. Chưa xác định nguyên nhân duy nhất nếu chưa có các ablation độc lập; không dùng kết quả này để tự đổi cấu hình và đánh giá lại test.", "",
              "## Kiểm chứng số học và nguồn gốc bảng dẫn xuất", "",
              "Verifier nghiêm ngặt dừng ở trường score_gap_top2_minus_top1: writer tính phép trừ float32, trong khi score matrix được lưu dưới dạng cast float64 và verifier tính phép trừ float64. Đây là trường chênh lệch score dẫn xuất, không phải nhãn dự đoán hoặc chỉ số accuracy.",
              f"Sai khác lớn nhất ở sáu run là {max(r['max_absolute_gap_rounding_error'] for r in gap_audit['runs']):.10g}; tất cả đều nằm trong giới hạn làm tròn float32 đã kiểm tra theo từng hàng.",
              "Không sửa file gốc. Bản verification_view riêng tính lại duy nhất gap bằng float64 từ score matrix gốc, tái sinh errors.csv dẫn xuất từ cùng prediction rồi chạy toàn bộ verifier có sẵn. Nhãn, thứ hạng, score matrix, metric, CI và lựa chọn model không thay đổi. Bảng evidence_verification.csv lưu cả hash prediction gốc và hash prediction của view để truy vết. CLI report nguyên bản trên raw float32 vẫn có thể dừng ở kiểm tra gap này; quy trình hậu xử lý được lưu riêng để bảo toàn source đã khóa.",
              "Bằng chứng chi tiết: outputs/cnn_experiments/float32_gap_verification_audit.json, verification_view/ và checkpoint_diagnostics.json.", "",
              "## Tổng hợp ba seed", "", "| Chỉ số | Mean ± sample std (điểm phần trăm) | Số seed |", "|---|---:|---:|"]
    for r in summary.itertuples(index=False):
        lines.append(f"| {r.metric} | {100*r.mean:.4f}% ± {100*r.sample_std_ddof1:.4f} | {r.n_seeds} |")
    lines += ["", "Std dùng ddof=1, mô tả biến thiên giữa ba lần train. Nó không phải CI của quần thể test. Mean là trung bình chỉ số của ba model riêng biệt; không phải accuracy của một ensemble hoặc một model được chọn bằng test.", "",
              "## So sánh trên cùng 6.000 probe", "",
              "| Pipeline | Rank-1 | Rank-5 | Macro-F1 |", "|---|---:|---:|---:|"]
    for r in classical.itertuples(index=False):
        lines.append(f"| {r.model} + CompCode | {100*r.rank1:.4f}% | {100*r.rank5:.4f}% | {100*r.macro_f1:.4f}% |")
    means = summary.set_index("metric")
    lines.append(f"| ResNet18 — mean 3 seed | {100*means.loc['rank1','mean']:.4f}% | "
                 f"{100*means.loc['rank5','mean']:.4f}% | {100*means.loc['macro_f1','mean']:.4f}% |")
    lines += ["", "So sánh chính dùng chênh lệch **mean Rank-1 của ba seed CNN − Rank-1 của baseline**. Mỗi lần bootstrap resample Palm ID một lần, áp dụng cùng probe cho tất cả model/seed; 5.000 lần, seed=42. Không gộp ba run thành 18.000 probe độc lập.", "",
              "| Baseline | Δ Rank-1 (điểm phần trăm) | CI 95% (điểm phần trăm) | Khoảng Bonferroni cho 3 so sánh, family-wise 95% |",
              "|---|---:|---|---|"]
    for r in paired.itertuples(index=False):
        lines.append(f"| {r.classical_model} | {100*r.delta_rank1_cnn_mean_minus_classical:.4f} | "
                     f"[{100*r.ci95_lower:.4f}; {100*r.ci95_upper:.4f}] | "
                     f"[{100*r.familywise95_bonferroni3_lower:.4f}; {100*r.familywise95_bonferroni3_upper:.4f}] |")
    lines += ["", "Khoảng bootstrap trên có điều kiện trên ba seed đã train; không bao hàm toàn bộ biến thiên của những seed tương lai. Các khoảng Bonferroni là khoảng biên 98,3333% cho từng so sánh để kiểm soát ba so sánh chính ở mức family-wise xấp xỉ 95%. Bảng so sánh từng seed với baseline được lưu riêng, mang tính mô tả và chưa điều chỉnh đa so sánh.", "",
              "## Độ trễ CPU cùng điều kiện từng ảnh", "",
              "| Model | Probe | Tổng median (ms) | Tổng P95 (ms) |", "|---|---:|---:|---:|"]
    for r in classical.itertuples(index=False):
        lines.append(f"| {r.model} + CompCode | {r.latency_n_queries} | {r.latency_median_ms:.4f} | {r.latency_p95_ms:.4f} |")
    for r in latency:
        lines.append(f"| {r['model']} | {r['n_queries']} | {r['median_ms']:.4f} | {r['p95_ms']:.4f} |")
    lines += ["", "CPU Ryzen 7 5800H, một luồng, model nạp sẵn, ba lượt warmup, một lượt đo toàn bộ probe cho mỗi model. CNN gồm đọc/giải mã ROI + transform + forward logits; classical gồm đọc/giải mã ROI + CompCode + score lớp. Không gồm model loading, sắp hạng Top-k hoặc UI. Bộ đệm file hệ điều hành có thể đã ấm. Accuracy CNN được đánh giá trên GPU; bảng này chỉ đo thời gian inference CPU của cùng checkpoint. CNN latency từng probe cũng được lưu CSV; thời gian đánh giá batch GPU được ghi riêng, không so trực tiếp với các cột CPU này.", "",
              "## Diễn giải và giới hạn", "",
              "- Đây là so sánh toàn pipeline: CNN học feature từ ROI, các model cổ điển dùng CompCode. Không quy toàn bộ chênh lệch cho riêng classifier.",
              "- Classical dùng 5-fold để chọn tham số; CNN dùng fold0 để chọn epoch. Cần ghi rõ khác biệt về chọn mô hình khi báo cáo.",
              "- Kết quả classical session2 đã được quan sát ở giai đoạn 1. Cấu hình CNN được lấy từ cấu hình sẵn có và chốt trước đánh giá CNN, nhưng toàn bộ nghiên cứu là so sánh tuần tự/exploratory, không phải một lần test hoàn toàn chưa từng được xem cho mọi candidate.",
              "- Bootstrap gom theo Palm ID. Thiếu ánh xạ Person ID để gom hai tay cùng chủ thể; chỉ có ba seed nên chưa ước lượng đầy đủ độ nhạy với mọi khởi tạo/hardware/backend.",
              "- Mã nguồn chứa warning kế thừa về pilot HOG, không có bằng chứng tương ứng; người thực hiện xác nhận chưa chạy model trước benchmark giai đoạn 1. Không dùng chuỗi warning đó làm lịch sử thực nghiệm đã được kiểm chứng.",
              "- CNN không dùng CompCode nên không phải một biến thể CR_CompCode. Kết quả của nhóm không được coi là tái hiện accuracy MATLAB tác giả.", "",
              "## Bằng chứng và tái lập", "",
              f"- Pretrained SHA-256: `{protocol['pretrained_weights']['sha256']}`.",
              f"- Manifest SHA-256: `{protocol['dataset']['manifest_sha256']}`.",
              f"- CNN source SHA-256: `{protocol['source_sha256']}`.",
              f"- Torch {protocol['environment']['torch']}, torchvision {protocol['environment']['torchvision']}, DirectML {protocol['environment']['torch_directml']}; dependency đầy đủ được lưu theo môi trường riêng.",
              "- Cấu hình, source snapshot, hash weights, log, tiến độ, test kỹ thuật: `outputs/cnn_experiments/`.",
              "- Khóa ba seed: `artifacts/cnn_protocol_lock.json`; trọng số/freeze: `artifacts/resnet18_seed17.*`, `seed29.*`, `seed43.*`.",
              "- Raw metric/score/prediction/confusion/CMC/per-class: `outputs/runs/resnet18_seed17/`, `resnet18_seed29/`, `resnet18_seed43/`.",
              "- Tổng hợp: `reports/tables/cnn_seed_results.csv`, `cnn_seed_summary.csv`, `cnn_mean_vs_classical_paired.csv`, `cnn_each_seed_vs_classical_paired.csv`, `cnn_cpu_latency.csv`.",
              "- Biểu đồ: `reports/figures/cnn_training_curves.png`, `cnn_vs_classical_cmc_rank20.png`; các biểu đồ CMC/confusion đầy đủ được sinh riêng cho từng run.",
              "- Các artifact và metric M1–M3 đã được kiểm tra hash, giữ nguyên so với đầu giai đoạn này.", "",
              "Lệnh đã chạy từ gốc dự án, trong môi trường DirectML có dependency snapshot:", "",
              "```powershell",
              "$env:TORCH_HOME = Join-Path (Get-Location) '.cache/torch'",
              ".runtime/venv-directml/Scripts/python.exe -u outputs/cnn_experiments/run_experiments.py",
              "# Resume postprocessing after the documented float32 score-gap verifier issue:",
              ".runtime/venv-directml/Scripts/python.exe -u outputs/cnn_experiments/checkpoint_diagnostics.py",
              ".runtime/venv-directml/Scripts/python.exe -u outputs/cnn_experiments/finish_experiments.py",
              ".runtime/venv-directml/Scripts/python.exe outputs/cnn_experiments/write_report.py",
              "```", "",
              "Runner giữ lại các seed đã freeze và kết quả test đã tồn tại; không ghi đè để chọn lại model. Nếu chạy một nghiên cứu khác, dùng workspace/protocol mới và công khai lịch sử sử dụng session2.", "",
              "## Nguồn tham khảo", "",
              "- [Torchvision ResNet18 và ImageNet weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).",
              "- [Transfer learning — hướng dẫn PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).",
              "- [PyTorch với DirectML trên Windows — Microsoft](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows).", ""]
    report = ws / "reports/CNN_EXPERIMENTS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if not any(event["stage"] == "postprocessing_and_cpu_latency_completed" for event in progress):
        progress.append({"stage": "postprocessing_and_cpu_latency_completed", "status": "completed",
                         "finished_at": done["completed_at"],
                         "note": "Original strict gap-verifier failure retained in history; resolved with audited derivative view"})
        atomic_json(progress_path, progress)
    for run in gap_audit["runs"]:
        name = run["model"]
        assert hash_file(ws / f"outputs/runs/{name}/predictions.csv") == run["raw_predictions_sha256"]
        assert hash_file(ws / f"outputs/runs/{name}/all_class_scores.npz") == run["raw_scores_sha256"]
    evidence = [report, *tables.glob("cnn_*.csv"), figures / "cnn_training_curves.png",
                figures / "cnn_vs_classical_cmc_rank20.png", out / "recognition_results.json",
                out / "run_complete.json", out / "checkpoint_diagnostics.json",
                out / "float32_gap_verification_audit.json", out / "progress.json",
                tables / "model_comparison.csv", tables / "evidence_verification.csv",
                ws / "artifacts/cnn_protocol_lock.json"]
    atomic_json(out / "final_integrity.json", {"classical_evidence_preserved": done["classical_evidence_preserved"],
                "source_unchanged_since_registration": result["source_unchanged_since_registration"],
                "sha256": {str(p.relative_to(ws)): hash_file(p) for p in evidence}})
    print(report)
    print(seeds[["seed", "rank1", "rank5", "macro_f1"]].to_string(index=False))
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()

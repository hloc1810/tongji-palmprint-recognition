# K?t qu? benchmark Tongji ? M1, M2, M3

Ho?n t?t: 2026-11-01T12:47:20.411493+07:00 (Asia/Saigon).

??y l? k?t qu? ?o th?c t? tr?n dataset c?c b? ?? qua strict audit; c?c s? trong b?ng ???c ??c t? artifact th?c nghi?m, kh?ng l?y t? smoke test ho?c b?i b?o.

## Giao th?c

- Closed-set 1:N, 600 Palm ID; session1 g?m 6.000 ?nh train/gallery, session2 g?m 6.000 ?nh probe.
- C?ng ROI grayscale 128?128 v? CompCode histogram 600 chi?u chu?n h?a L2 cho c? ba m? h?nh.
- 5-fold session1, seed=42: 8 train + 2 validation ?nh/Palm ID/fold; kh?ng ch?n tham s? b?ng session2.
- Grid SVM C={0,1; 1; 10}; CRC ?={0,1; 1,35; 10}; ch?n mean CV Rank-1 l?n nh?t, h?a th? ch?n tham s? nh? nh?t.
- ?? fit l?i tr?n to?n session1 v? kh?a c? ba artifact tr??c khi tr?ch feature/??nh gi? nh?n d?ng session2.
- CompCode l? b?n paper-inspired; ch?a ki?m ch?ng parity s? h?c v?i MATLAB t?c gi?. CRC d?ng residual Euclidean kh?ng chia cho norm h? s? l?p.

## K?t qu? nh?n d?ng cross-session

| M? h?nh | Tham s? | Mean CV Rank-1 | Test Rank-1 | Test Rank-5 | Macro-F1 | ??ng / 6.000 | CI 95% Rank-1 |
|---|---|---:|---:|---:|---:|---:|---|
| M1 ? 1-NN nearest-template | k = 1 | 99.9833% | 94.4333% | 96.9667% | 93.5583% | 5666 | [93.0167%; 95.6842%] |
| M2 ? Linear SVM OvR | C = 1 | 100.0000% | 97.3667% | 98.5167% | 96.8154% | 5842 | [96.1667%; 98.3000%] |
| M3 ? CRC / Collaborative Ridge | ? = 0,1 | 100.0000% | 98.2667% | 99.0833% | 97.9926% | 5896 | [97.4000%; 98.9333%] |

CI d?ng bootstrap theo Palm ID, 1.000 l?n l?p, seed=42. Kh?ng c? ?nh x? Person ID ?? gom hai tay c?ng ng??i. Macro-F1 trong b?ng hi?n th? theo thang ph?n tr?m; CSV l?u t? l? 0?1.

SVM C=1 v? C=10 c?ng ??t 100% CV n?n ch?n C=1. C? ba ? CRC c?ng ??t 100% CV n?n ch?n ?=0,1 theo tie-break ?? ??ng k?. ?i?u n?y kh?ng ch?ng minh ?=0,1 l? t?t nh?t tr?n m?i mi?n d? li?u.

## ?? tr? CPU t?ng ?nh

Ph?n c?ng: AMD Ryzen 7 5800H with Radeon Graphics; RAM 15.38 GiB; Windows; m?t lu?ng cho OpenCV v? BLAS. M?i m? h?nh ?o m?t l??t tr?n to?n b? 6.000 probe, theo th? t? manifest, ch?y l?n l??t t?ng m? h?nh.

| M? h?nh | ??c ROI median (ms) | CompCode median (ms) | Ch?m ?i?m median (ms) | T?ng median (ms) | T?ng P95 (ms) |
|---|---:|---:|---:|---:|---:|
| M1 ? 1-NN nearest-template | 0.5996 | 2.9940 | 7.3576 | 10.9987 | 20.2279 |
| M2 ? Linear SVM OvR | 0.5271 | 2.9419 | 0.2248 | 3.7102 | 5.5744 |
| M3 ? CRC / Collaborative Ridge | 0.6892 | 3.0671 | 10.9071 | 14.7064 | 20.8129 |

T?ng g?m ??c/gi?i m? ROI + tr?ch CompCode + t?nh score cho ?? 600 l?p; kh?ng g?m n?p model, s?p h?ng Top-k ho?c hi?n th? giao di?n. Model ?? ???c n?p s?n; c? 3 l??t warmup matching. Feature ???c t?nh l?i cho t?ng ?nh; b? ??m file c?a h? ?i?u h?nh c? th? ?? ?m. Median t?ng ???c t?nh tr?n t?ng th?i gian t?ng ?nh, kh?ng c?ng c?c median th?nh ph?n.

Th?i gian classifier x? l? batch/chia b?nh qu?n ???c l?u ri?ng trong CSV; kh?ng d?ng thay cho ?? tr? t?ng ?nh ? b?ng n?y.

## So s?nh c? gh?p c?p

| So s?nh Rank-1 | Ch?nh l?ch (?i?m ph?n tr?m) | CI 95% (?i?m ph?n tr?m) |
|---|---:|---|
| crc ? knn | 3.8333 | [2.8333; 4.8833] |
| crc ? svm | 0.9000 | [0.4000; 1.5000] |
| knn ? svm | -2.9333 | [-3.7671; -2.0333] |

Bootstrap gh?p c?p gi? nguy?n c?ng probe v? gom theo Palm ID. CRC cao h?n SVM 0,90 ?i?m ph?n tr?m trong l?n ?o n?y, v?i CI gh?p c?p [0,40; 1,50]. SVM c? median ?? tr? th?p h?n CRC trong ?i?u ki?n ?o ?? ghi.

## Th?i gian fit final v? k?ch th??c artifact

| M? h?nh | Fit final (s) | Artifact (MiB) |
|---|---:|---:|
| M1 ? 1-NN nearest-template | 0.007858 | 6.0005 |
| M2 ? Linear SVM OvR | 45.583415 | 2.6465 |
| M3 ? CRC / Collaborative Ridge | 0.076052 | 15.5866 |

Fit final kh?ng g?m tr?ch feature, CV, ki?m tra hash ho?c ghi artifact. V?i 1-NN, fit l? d?ng gallery; v?i CRC, fit l? d?ng dictionary v? factor h?a h? ridge.

## H? s? t?i l?p v? ki?m ch?ng

- Manifest SHA-256: `99bbbe45a4807d48801a28a6aadfec2efee46a3f4c4fe3d3af8e2337241c8ae0`.
- Classical source SHA-256: `4e324e0b8840cb473dcda4cfcf75c3344b21f64d03360a2efe0bb502ad3b05bc`; ?? x?c nh?n kh?ng ??i trong l?n ch?y.
- M?i tr??ng: Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, OpenCV 5.0.0, scikit-image 0.26.0.
- B?ng ??y ??: `reports/tables/tongji_benchmark.csv`; l??i CV: `reports/tables/tongji_cv_grid_summary.csv`.
- Ki?m ch?ng scores/predictions/metrics: `reports/tables/evidence_verification.csv`; c? ba run ?? ??t.
- Raw evidence: `outputs/runs/knn/`, `outputs/runs/svm/`, `outputs/runs/crc/`: metrics, predictions, all-class scores, confusion, per-class, CMC, errors v? CPU benchmark.
- Bi?u ?? CMC/confusion/CI v? so s?nh: `reports/figures/`.
- C?u h?nh tr??c test, source snapshot, dependencies, hardware, log v? ti?n ??: `outputs/tongji_benchmark/`.
- Artifact v? kh?a giao th?c: `artifacts/knn.joblib`, `svm.joblib`, `crc.joblib`, c?c freeze JSON v? `protocol_lock.json`.

L?u ? v? l?ch s?: raw metrics c?n tr??ng warning v? m?t pilot HOG c? v? m? ngu?n ban ??u ch?a chu?i ??. Kh?ng c? b?ng ch?ng th?c nghi?m t??ng ?ng trong workspace; ng??i th?c hi?n x?c nh?n ch?a ch?y m? h?nh tr??c l?n n?y. Warning n?y l? n?i dung k? th?a, kh?ng ph?i k?t lu?n r?t ra t? d? li?u. M? thu?t to?n ???c gi? nguy?n khi ch?y.

K?t qu? ch? ?p d?ng cho l?n ?o, giao th?c v? b?n tri?n khai n?y. ?i?m CV g?n 100% kh?ng thay th? ?i?m test cross-session. Ch?a th? g?i ??y l? t?i hi?n CR_CompCode nguy?n b?n c?a t?c gi?.

## Ngu?n ph??ng ph?p

- [Trang dataset v? ph??ng ph?p c?a t?c gi? Tongji](https://cslinzhang.github.io/ContactlessPalm/).
- [B?i b?o Tongji, Pattern Recognition 69 (2017), ?5](https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf).
- [LinearSVC ? t?i li?u scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

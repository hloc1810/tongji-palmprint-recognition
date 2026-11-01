import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { Workbook } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const tables = path.join(root, 'reports/tables');
const seeds = [17, 29, 43];
const readJSON = async p => JSON.parse(await fs.readFile(path.join(root, p), 'utf8'));
const hash = async p => crypto.createHash('sha256').update(await fs.readFile(path.join(root, p))).digest('hex');
const verified = await readJSON('outputs/embedding_benchmark/verification.json');
assert.equal(verified.runs.length, 6);
assert(verified.runs.every(r => r.verified));

async function readCSV(name) {
  const text = await fs.readFile(path.join(tables, name), 'utf8');
  const book = await Workbook.fromCSV(text, {sheetName: 'Results'});
  const values = book.worksheets.getItem('Results').getUsedRange().values;
  const header = values[0];
  const rows = values.slice(1).filter(r => r[0] !== null && r[0] !== '').map(r =>
    Object.fromEntries(header.map((h, i) => [h, r[i] ?? ''])));
  return {text, book, header, rows};
}

const main = await readCSV('tongji_benchmark.csv');
const comparison = await readCSV('model_comparison.csv');
const cpu = await readCSV('cnn_cpu_latency.csv');
assert.equal(main.rows.length, 3, 'Expected original classical-only table');
assert.equal(comparison.rows.length, 6, 'Expected original classical + logits runs');

// Back up only the report files being extended. Raw evidence is read-only.
const targets = ['reports/tables/tongji_benchmark.csv', 'reports/tables/model_comparison.csv',
                 'reports/TONGJI_BENCHMARK.md', 'reports/CNN_EXPERIMENTS.md'];
await fs.mkdir(path.join(here, 'before'), {recursive: true});
for (const name of targets) {
  const dest = path.join(here, 'before', path.basename(name));
  await fs.copyFile(path.join(root, name), dest, fs.constants.COPYFILE_EXCL);
}

// A CSV has no formatting. Preview the existing key columns before extending it.
const previewStyle = book => {
  const s = book.worksheets.getItem('Results');
  s.getRange('A1:F13').format.columnWidth = 22;
  s.getRange('A1:A13').format.columnWidth = 51;
  s.getRange('A1:F13').format.font = {name: 'Arial', size: 10};
};
previewStyle(main.book);
const before = await main.book.render({sheetName: 'Results', range: 'A1:F4', scale: 1, format: 'png'});
await fs.writeFile(path.join(here, 'before.png'), new Uint8Array(await before.arrayBuffer()));

const extras = ['seed', 'gallery_strategy', 'feature_representation', 'embedding_dimension',
  'n_gallery', 'gallery_sha256', 'evaluation_stage', 'metrics_source',
  'probe_feature_ms_per_image_batch_amortized', 'cosine_matching_ms_per_image_batch_amortized',
  'batch_timing_scope'];
const mainHeader = [...new Set([...main.header, ...extras])];
const compareHeader = [...new Set([...comparison.header, ...extras])];
const baselineRuns = [];
const newRows = [];
for (const seed of seeds) {
  const runName = `resnet18_seed${seed}`;
  const baseline = await readJSON(`outputs/runs/${runName}/metrics.json`);
  baselineRuns.push(baseline);
  const old = comparison.rows.find(r => r.model === runName);
  const timing = cpu.rows.find(r => r.model === runName);
  main.rows.push({...old, seed, feature_representation: 'ResNet18 512-D + 600-class FC logits',
    correct_probes: Math.round(baseline.rank1 * baseline.n), latency_n_queries: timing.n_queries,
    latency_median_ms: timing.median_ms, latency_p95_ms: timing.p95_ms,
    artifact_size_bytes: (await fs.stat(path.join(root, `artifacts/${runName}.pth`))).size,
    evaluation_stage: 'original CNN experiment', metrics_source: `outputs/runs/${runName}/metrics.json`});
  for (const strategy of ['nearest', 'centroid']) {
    const model = `${runName}_embedding_${strategy}`;
    const source = `outputs/runs/${model}/metrics.json`;
    const m = await readJSON(source);
    assert.equal(m.artifact_sha256, baseline.artifact_sha256);
    const row = {model, rank1: m.rank1, rank5: m.rank5,
      macro_precision: m.macro_precision, macro_recall: m.macro_recall, macro_f1: m.macro_f1,
      rank1_ci_lower: m.rank1_ci95_palm_bootstrap[0], rank1_ci_upper: m.rank1_ci95_palm_bootstrap[1],
      matching_ms_per_probe_batch_amortized: m.cosine_matching_ms_per_image_batch_amortized,
      n: m.n, score_kind: m.score_kind, method: m.method, data_scope: 'strict_tongji_layout',
      manifest_sha256: m.manifest_sha256, artifact_sha256: m.artifact_sha256,
      note: 'Exploratory post-hoc; same frozen CNN checkpoint. CPU batch-amortized timings; no single-image latency measurement. No new model training.',
      correct_probes: Math.round(m.rank1 * m.n), seed, gallery_strategy: strategy,
      feature_representation: 'ResNet18 512-D L2-normalized', embedding_dimension: m.embedding_dimension,
      n_gallery: m.n_gallery, gallery_sha256: m.gallery_sha256,
      evaluation_stage: 'exploratory post-hoc', metrics_source: source,
      probe_feature_ms_per_image_batch_amortized: m.probe_feature_ms_per_image_batch_amortized,
      cosine_matching_ms_per_image_batch_amortized: m.cosine_matching_ms_per_image_batch_amortized,
      batch_timing_scope: 'CPU batch: cosine matching only; probe feature extraction recorded separately'};
    newRows.push(row);
  }
}
for (const row of main.rows.slice(0, 3)) {
  row.feature_representation = 'CompCode histogram 600-D L2-normalized';
  row.evaluation_stage = 'original classical experiment';
  row.metrics_source = `outputs/runs/${row.model}/metrics.json`;
  row.batch_timing_scope = 'classical batch: classifier scores only, CPU';
}
main.rows.push(...newRows);
comparison.rows.push(...newRows);

const mean = x => x.reduce((a, b) => a + Number(b), 0) / x.length;
const std = x => Math.sqrt(x.reduce((a, b) => a + (Number(b) - mean(x)) ** 2, 0) / (x.length - 1));
const summary = [];
const groups = [
  ['M1', '1-NN nearest-template + CompCode', [main.rows[0]], 'original classical experiment'],
  ['M2', 'Linear SVM OvR + CompCode', [main.rows[1]], 'original classical experiment'],
  ['M3', 'CRC_RLS / Collaborative Ridge + CompCode', [main.rows[2]], 'original classical experiment'],
  ['M4-E1', 'ResNet18 600-class logits', baselineRuns, 'original CNN experiment'],
  ['M4-E2-nearest', 'ResNet18 512-D Nearest Cosine', newRows.filter(r => r.gallery_strategy === 'nearest'), 'exploratory post-hoc'],
  ['M4-E2-centroid', 'ResNet18 512-D Centroid Cosine', newRows.filter(r => r.gallery_strategy === 'centroid'), 'exploratory post-hoc'],
];
const summaryHeader = ['pipeline', 'method', 'n_runs', 'seeds', 'classes', 'n_probes_per_run',
  'rank1_mean', 'rank1_sample_std_ddof1', 'rank5_mean', 'rank5_sample_std_ddof1',
  'macro_f1_mean', 'macro_f1_sample_std_ddof1', 'rank1_delta_vs_logits_pp', 'evaluation_stage'];
for (const [pipeline, method, runs, stage] of groups) {
  const row = {pipeline, method, n_runs: runs.length, seeds: runs.length === 3 ? seeds.join(';') : '',
    classes: 600, n_probes_per_run: 6000, evaluation_stage: stage};
  for (const metric of ['rank1', 'rank5', 'macro_f1']) {
    row[`${metric}_mean`] = mean(runs.map(r => r[metric]));
    row[`${metric}_sample_std_ddof1`] = runs.length > 1 ? std(runs.map(r => r[metric])) : '';
  }
  row.rank1_delta_vs_logits_pp = pipeline.startsWith('M4-E2')
    ? 100 * (row.rank1_mean - mean(baselineRuns.map(r => r.rank1))) : '';
  summary.push(row);
}

const quote = v => {
  const s = v === null || v === undefined ? '' : String(v);
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
};
async function exportCSV(name, header, rows) {
  const book = Workbook.create();
  const sheet = book.worksheets.add('Results');
  sheet.getRangeByIndexes(0, 0, rows.length + 1, header.length).values =
    [header, ...rows.map(r => header.map(h => r[h] ?? ''))];
  book.recalculate();
  const values = sheet.getUsedRange().values;
  assert.equal(values.length, rows.length + 1);
  // Plain CSV export preserves the source precision and contains no Excel-only features.
  await fs.writeFile(path.join(tables, name), values.map(r => r.map(quote).join(',')).join('\n') + '\n');
  return book;
}
const finalBook = await exportCSV('tongji_benchmark.csv', mainHeader, main.rows);
await exportCSV('model_comparison.csv', compareHeader, comparison.rows);
await exportCSV('tongji_benchmark_summary.csv', summaryHeader, summary);
previewStyle(finalBook);
const after = await finalBook.render({sheetName: 'Results', range: 'A1:F13', scale: 1, format: 'png'});
await fs.writeFile(path.join(here, 'after.png'), new Uint8Array(await after.arrayBuffer()));

const pct = n => (100 * Number(n)).toFixed(4);
const display = (r, metric) => `${pct(r[`${metric}_mean`])}%${r.n_runs > 1 ? ` ± ${pct(r[`${metric}_sample_std_ddof1`])}` : ''}`;
const summaryLines = summary.map(r => `| ${r.pipeline} — ${r.method} | ${display(r, 'rank1')} | ${display(r, 'rank5')} | ${display(r, 'macro_f1')} |`);
const perSeed = newRows.map(r => `| ${r.seed} | ${r.gallery_strategy === 'nearest' ? 'Nearest Cosine' : 'Centroid Cosine'} | ${pct(r.rank1)}% | ${pct(r.rank5)}% | ${pct(r.macro_f1)}% | ${r.correct_probes} | [${pct(r.rank1_ci_lower)}%; ${pct(r.rank1_ci_upper)}%] |`);
const timeLines = newRows.map(r => `| ${r.seed} | ${r.gallery_strategy} | ${Number(r.probe_feature_ms_per_image_batch_amortized).toFixed(4)} | ${Number(r.cosine_matching_ms_per_image_batch_amortized).toFixed(4)} |`);
const section = `# Benchmark Tongji: mô hình cổ điển, CNN và embedding matching

Closed-set 1:N, 600 Palm ID. Mỗi run dùng 6.000 ảnh session1 để train/gallery và 6.000 probe session2. Các hàng CNN tổng hợp ba seed 17, 29, 43. Kết quả embedding được đọc từ metrics và đã đối chiếu score, prediction, manifest, hash checkpoint và gallery của cả sáu run.

## Bảng tổng hợp các pipeline

| Pipeline | Rank-1 | Rank-5 | Macro-F1 |
|---|---:|---:|---:|
${summaryLines.join('\n')}

Các giá trị sau dấu ± là sample std giữa ba seed, ddof=1, theo điểm phần trăm. Đây là trung bình chỉ số của các model riêng biệt, không phải ensemble hoặc 18.000 probe độc lập. Các hàng cổ điển là một cấu hình final mỗi mô hình. CSV dùng tỷ lệ 0–1; cột delta có đơn vị điểm phần trăm.

Nearest Cosine tăng mean Rank-1 ${Number(summary[4].rank1_delta_vs_logits_pp).toFixed(4)} điểm phần trăm so với logits. Centroid Cosine tăng ${Number(summary[5].rank1_delta_vs_logits_pp).toFixed(4)} điểm phần trăm. Đây là chênh lệch mô tả, chưa phải kết luận có ý nghĩa thống kê từ kiểm định ghép cặp mới.

## Embedding matching: kết quả từng seed

ResNet18 → feature 512-D → L2 normalization → gallery cosine. Dùng lại checkpoint của M4-E1 cùng seed, không train thêm. Nearest dùng cosine lớn nhất trong 10 template của mỗi Palm ID. Centroid lấy trung bình embedding đã chuẩn hóa của lớp rồi chuẩn hóa lại, tạo 600 centroid từ 6.000 ảnh gallery.

| Seed | Kỹ thuật | Rank-1 | Rank-5 | Macro-F1 | Đúng / 6.000 | CI 95% Rank-1 |
|---|---|---:|---:|---:|---:|---|
${perSeed.join('\n')}

CI từng run là bootstrap theo Palm ID theo cấu hình evaluator. Không lấy trung bình hai cận CI để tạo CI cho mean ba seed. Hai kỹ thuật embedding được bổ sung sau khi đã xem kết quả session2, nên được ghi là **so sánh hậu nghiệm/exploratory**. Không dùng session2 để chọn seed hoặc tuyên bố chiến lược đã được chọn bằng validation. Chuỗi warning trong raw metrics nhắc “historical pilots” là nội dung kế thừa; bảng này chỉ khẳng định các kết quả session2 trước đó đã được quan sát.

## Thời gian embedding trên CPU theo batch

| Seed | Kỹ thuật | Trích feature probe (ms/ảnh, batch-amortized) | Cosine matching (ms/ảnh, batch-amortized) |
|---|---|---:|---:|
${timeLines.join('\n')}

Feature probe được trích một lần mỗi seed và dùng chung cho hai chiến lược. Các giá trị trên là thời gian cả batch chia số ảnh, không phải median/P95 từng ảnh và không so trực tiếp với bảng độ trễ CPU cũ. Gallery đã được tạo sẵn; gallery_wall_ms trong raw metrics của lượt evaluate là thời gian nạp/kiểm tra cache, không phải chi phí tạo gallery lần đầu. Chưa có phép đo độ trễ từng ảnh cho embedding, các cột tương ứng được để trống.

## File kết quả

- [Từng run: 3 classical + 3 logits + 6 embedding](tables/tongji_benchmark.csv).
- [Mean và sample std theo pipeline](tables/tongji_benchmark_summary.csv).
- [Bảng so sánh từng run](tables/model_comparison.csv).
- [Thiết kế và kết quả CNN ban đầu](CNN_EXPERIMENTS.md).
- [Định nghĩa embedding matching](../docs/EMBEDDING_MATCHING.md).

Nguồn của mỗi hàng mới nằm ở cột metrics_source, kèm seed, strategy, hash artifact/gallery và phạm vi timing. Kiểm chứng sáu run: outputs/embedding_benchmark/verification.json. Các file raw kết quả và model được giữ nguyên.

---

`;
const oldReport = await fs.readFile(path.join(here, 'before/TONGJI_BENCHMARK.md'), 'utf8');
await fs.writeFile(path.join(root, 'reports/TONGJI_BENCHMARK.md'), section + oldReport);
const cnnReport = await fs.readFile(path.join(here, 'before/CNN_EXPERIMENTS.md'), 'utf8');
const addition = `
## Bổ sung hậu nghiệm: embedding cosine từ cùng checkpoint

Đã đánh giá cả ba seed 17, 29, 43 trên cùng 6.000 probe session2 với gallery 6.000 ảnh session1. Không train thêm hoặc thay thế kết quả logits ở trên.

| Biến thể | Rank-1 mean ± std | Rank-5 mean ± std | Macro-F1 mean ± std |
|---|---:|---:|---:|
${summary.slice(4).map(r => `| ${r.method} | ${display(r, 'rank1')} | ${display(r, 'rank5')} | ${display(r, 'macro_f1')} |`).join('\n')}

Std dùng ddof=1, đơn vị điểm phần trăm. Đây là so sánh exploratory sau khi session2 đã được quan sát. Các CI ghép cặp và biểu đồ ở phần trước chỉ áp dụng cho logits M4-E1 ban đầu. [Bảng benchmark cập nhật](TONGJI_BENCHMARK.md) lưu từng seed, CI từng run và phạm vi thời gian xử lý batch của embedding.
`;
await fs.writeFile(path.join(root, 'reports/CNN_EXPERIMENTS.md'), cnnReport + addition);

// Confirm preservation of all original cells and experiment evidence.
for (const [file, beforeFile] of [['tongji_benchmark.csv', 'tongji_benchmark.csv'], ['model_comparison.csv', 'model_comparison.csv']]) {
  const oldBook = await Workbook.fromCSV(await fs.readFile(path.join(here, 'before', beforeFile), 'utf8'), {sheetName:'Results'});
  const oldValues = oldBook.worksheets.getItem('Results').getUsedRange().values;
  const current = await readCSV(file);
  const currentValues = current.book.worksheets.getItem('Results').getUsedRange().values;
  for (let i = 0; i < oldValues.length; i++)
    for (let j = 0; j < oldValues[0].length; j++) assert.equal(currentValues[i][j], oldValues[i][j]);
  assert.equal(current.rows.length, 12);
  assert.equal(new Set(current.rows.map(r => r.model)).size, 12);
}
for (const [p, expected] of Object.entries(verified.protected_sha256)) assert.equal(await hash(p), expected, p);
assert.equal((await readCSV('tongji_benchmark_summary.csv')).rows.length, 6);
await fs.writeFile(path.join(here, 'completion.json'), JSON.stringify({
  completed: true, existing_table_values_preserved: true, source_and_raw_evidence_preserved: true,
  verified_embedding_runs: 6, total_runs_per_detail_table: 12, pipelines_in_summary: 6,
  outputs: [...targets, 'reports/tables/tongji_benchmark_summary.csv'], summary
}, null, 2));
console.log(JSON.stringify(summary, null, 2));

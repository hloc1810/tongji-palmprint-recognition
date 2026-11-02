"""Update Tongji_Palmprint_UI.html to connect with real project models.

Replaces simulation and mock data with live API inference, real Session 2 probe samples,
model selector (CRC_RLS, Linear SVM, 1-NN), top-5 ranking candidates, and real CPU latencies.
Preserves exact layout, fonts, Inter variable font subset, styling, and design tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
HTML_PATH = WS / "Tongji_Palmprint_UI.html"
SAMPLES_CACHE = WS / "app/samples_cache.json"


def update_ui():
    html = HTML_PATH.read_text(encoding="utf-8")
    samples = json.loads(SAMPLES_CACHE.read_text(encoding="utf-8"))

    # 1. Add CSS rules for model selector, sample grid, and top-5 candidates
    css_target = "    /* ==========================================================================\n       Toast\n       ========================================================================== */"
    new_css = (
        """    /* ---- Chọn mô hình & Chọn mẫu probe ---- */
    .chip-success{background:var(--accent-soft);border-color:var(--accent-line);color:var(--accent-text);font-weight:650;letter-spacing:.04em;border-radius:var(--r-chip);height:24px;padding:0 8px}
    .chip-danger{background:var(--danger-bg);border-color:var(--danger);color:var(--danger);font-weight:650;letter-spacing:.04em;border-radius:var(--r-chip);height:24px;padding:0 8px}

    .model-group{margin:10px 0 4px}
    .model-select{width:100%;height:36px;padding:0 10px;font-family:var(--font);font-size:13px;font-weight:500;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:var(--r-ctl);outline:none;transition:border-color .15s}
    .model-select:focus{border-color:var(--accent)}

    .sample-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:8px 0 10px}
    .sample-chip{display:flex;align-items:center;justify-content:center;height:28px;padding:0 6px;font-size:11.5px;font-weight:500;background:var(--surface-2);border:1px solid var(--line);border-radius:var(--r-chip);cursor:pointer;color:var(--ink-2);transition:all .15s;white-space:nowrap}
    .sample-chip:hover,.sample-chip.active{background:var(--accent-soft);border-color:var(--accent-line);color:var(--accent-text);font-weight:650}

    /* ---- Top 5 ứng viên ---- */
    .top5-wrap{margin-top:14px;padding-top:12px;border-top:1px dashed var(--line)}
    .top5-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3);margin-bottom:8px}
    .top5-row{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}
    .top5-item{padding:6px 2px;background:var(--surface-2);border:1px solid var(--line);border-radius:var(--r-chip);text-align:center;min-width:0}
    .top5-item.top1{background:var(--accent-soft);border-color:var(--accent-line)}
    .top5-rank{font-size:10px;color:var(--ink-3);font-weight:600}
    .top5-id{font-size:12px;font-weight:700;color:var(--ink);margin:1px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .top5-score{font-size:10px;font-family:var(--mono);color:var(--ink-2)}

"""
        + css_target
    )

    if css_target in html and ".model-group" not in html:
        html = html.replace(css_target, new_css, 1)

    # 2. Update topbar status pill
    old_pill = '<div class="status-pill pill-demo" id="modePill"><span class="dot"></span><span id="modeText">Chế độ minh họa</span></div>'
    new_pill = '<div class="status-pill" id="modePill"><span class="dot"></span><span id="modeText">Kết nối AI trực tiếp</span></div>'
    if old_pill in html:
        html = html.replace(old_pill, new_pill, 1)

    # 3. Update input-card pane
    old_input_actions = """                <div class="input-actions">
                  <div class="step-caption">Bắt đầu nhận dạng</div>
                  <button class="btn" type="button" id="uploadButton"><svg class="ico"><use href="#i-upload"/></svg>Tải ảnh lên</button>
                  <button class="btn" type="button" id="sampleButton"><svg class="ico"><use href="#i-image"/></svg>Dùng ảnh mẫu</button>
                  <div class="file-line"><svg class="ico ico-sm"><use href="#i-file"/></svg><span id="filename">palm-roi-demo-128.png</span></div>
                  <button type="button" class="btn btn-primary btn-lg action-run" id="recognizeButton"><span id="recognizeLabel">Thử nhận dạng demo</span><svg class="ico"><use href="#i-arrow"/></svg></button>
                  <div class="helper-line"><svg class="ico ico-sm"><use href="#i-info"/></svg><span>PNG, JPG, WebP · ROI 128×128</span></div>
                </div>"""

    new_input_actions = """                <div class="input-actions">
                  <div class="step-caption">Nguồn ảnh đầu vào</div>
                  <button class="btn" type="button" id="uploadButton"><svg class="ico"><use href="#i-upload"/></svg>Tải ảnh lên</button>
                  <button class="btn" type="button" id="sampleButton"><svg class="ico"><use href="#i-image"/></svg>Đổi ảnh mẫu tiếp theo</button>

                  <div class="step-caption" style="margin-top:6px">Ảnh mẫu Session 2 (Thử nghiệm chéo phiên):</div>
                  <div class="sample-grid" id="sampleGrid">
                    <button type="button" class="sample-chip active" data-sample-idx="0">Mẫu 1 · ID 0001</button>
                    <button type="button" class="sample-chip" data-sample-idx="1">Mẫu 2 · ID 0042</button>
                    <button type="button" class="sample-chip" data-sample-idx="2">Mẫu 3 · ID 0128</button>
                    <button type="button" class="sample-chip" data-sample-idx="3">Mẫu 4 · ID 0300</button>
                    <button type="button" class="sample-chip" data-sample-idx="4">Mẫu 5 · ID 0427</button>
                    <button type="button" class="sample-chip" data-sample-idx="5">Mẫu 6 · ID 0599</button>
                  </div>

                  <div class="model-group">
                    <div class="step-caption">Thuật toán nhận dạng (Đã khóa benchmark):</div>
                    <select class="model-select" id="modelSelect" aria-label="Thuật toán nhận dạng">
                      <option value="crc" selected>CRC_RLS (M3 · Rank-1 = 98.27% · Đề xuất)</option>
                      <option value="svm">Linear SVM (M2 · Rank-1 = 97.37% · Nhanh 3.71 ms)</option>
                      <option value="knn">1-NN Baseline (M1 · Rank-1 = 94.43% · k=1)</option>
                    </select>
                  </div>

                  <div class="file-line"><svg class="ico ico-sm"><use href="#i-file"/></svg><span id="filename">session2_00001.bmp</span></div>
                  <button type="button" class="btn btn-primary btn-lg action-run" id="recognizeButton"><span id="recognizeLabel">Nhận dạng bằng AI</span><svg class="ico"><use href="#i-arrow"/></svg></button>
                  <div class="helper-line"><svg class="ico ico-sm"><use href="#i-info"/></svg><span>PNG, JPG, WebP, BMP · ROI 128×128 · CompCode 600-D</span></div>
                </div>"""

    if old_input_actions in html:
        html = html.replace(old_input_actions, new_input_actions, 1)

    # 4. Update result-card pane
    old_result_card = """            <section class="pane result-card is-empty" id="resultCard" aria-live="polite">
              <div class="card-head"><div class="card-title"><svg class="ico"><use href="#i-check"/></svg>Kết quả nhận dạng</div><span class="chip chip-warn result-sim" id="resultBadge">MÔ PHỎNG</span></div>
              <div class="result-main">
                <div class="result-art" id="resultArt"><svg class="ico"><use href="#i-hand"/></svg></div>
                <div class="result-data">
                  <div class="minor-label" id="identityLabel">Mã lòng bàn tay</div>
                  <strong class="identity" id="identityValue">Chưa có kết quả</strong>
                  <div class="match-caption" id="resultDescription">Nhấn nút nhận dạng để xem kết quả.</div>
                  <div class="ok-stat" id="resultExtra"><svg class="ico"><use href="#i-shield"/></svg>Hỗ trợ 600 mã lòng bàn tay</div>
                </div>
              </div>
              <div class="result-info">
                <div class="metric-box"><div class="name"><svg class="ico"><use href="#i-chart"/></svg><span id="metricName">Điểm so khớp</span></div><div class="value" id="metricScore">—</div></div>
                <div class="metric-box"><div class="name"><svg class="ico"><use href="#i-clock"/></svg><span>Thời gian xử lý</span></div><div class="value" id="metricTime">—</div></div>
              </div>
              <div class="result-hint"><svg class="ico"><use href="#i-info"/></svg><span id="resultHint">Dữ liệu demo chỉ minh họa giao diện, không phải dự đoán của AI.</span></div>
            </section>"""

    new_result_card = """            <section class="pane result-card is-empty" id="resultCard" aria-live="polite">
              <div class="card-head"><div class="card-title"><svg class="ico"><use href="#i-check"/></svg>Kết quả nhận dạng</div><span class="chip chip-success" id="resultBadge">AI THẬT</span></div>
              <div class="result-main">
                <div class="result-art" id="resultArt"><svg class="ico"><use href="#i-hand"/></svg></div>
                <div class="result-data">
                  <div class="minor-label" id="identityLabel">Mã lòng bàn tay dự đoán</div>
                  <strong class="identity" id="identityValue">Chưa có kết quả</strong>
                  <div class="match-caption" id="resultDescription">Nhấn nút nhận dạng để xem kết quả.</div>
                  <div class="ok-stat" id="resultExtra"><svg class="ico"><use href="#i-shield"/></svg><span id="resultExtraText">Hỗ trợ 600 mã lòng bàn tay</span></div>
                </div>
              </div>
              <div class="result-info">
                <div class="metric-box"><div class="name"><svg class="ico"><use href="#i-chart"/></svg><span id="metricName">Điểm so khớp</span></div><div class="value" id="metricScore">—</div></div>
                <div class="metric-box"><div class="name"><svg class="ico"><use href="#i-clock"/></svg><span>Thời gian xử lý</span></div><div class="value" id="metricTime">—</div></div>
              </div>
              <div class="top5-wrap" id="top5Wrap" hidden>
                <div class="top5-title">Top 5 ứng viên tương đồng nhất:</div>
                <div class="top5-row" id="top5Row"></div>
              </div>
              <div class="result-hint"><svg class="ico"><use href="#i-info"/></svg><span id="resultHint">Kết quả suy luận trực tiếp từ checkpoint đã khóa của đề tài.</span></div>
            </section>"""

    if old_result_card in html:
        html = html.replace(old_result_card, new_result_card, 1)

    # 5. Update history table header to show model
    old_recent_thead = "<th>Ảnh</th><th>Thời gian</th><th>Mã nhận dạng</th><th>Nguồn</th>"
    new_recent_thead = (
        "<th>Ảnh</th><th>Thời gian</th><th>Mã nhận dạng</th><th>Mô hình</th><th>Điểm / Độ trễ</th>"
    )
    if old_recent_thead in html:
        html = html.replace(old_recent_thead, new_recent_thead, 1)

    old_hist_thead = "<th>STT</th><th>Thời gian</th><th>Mã lòng bàn tay</th><th>Trạng thái</th>"
    new_hist_thead = "<th>STT</th><th>Thời gian</th><th>Mã lòng bàn tay</th><th>Mô hình</th><th>Điểm số</th><th>Độ trễ</th><th>Trạng thái</th>"
    if old_hist_thead in html:
        html = html.replace(old_hist_thead, new_hist_thead, 1)

    # 6. Replace JavaScript script completely
    script_start = "  <script>\n  (function(){\n    'use strict';"
    script_end = "  })();\n  </script>"

    idx1 = html.find(script_start)
    idx2 = html.find(script_end)
    if idx1 == -1 or idx2 == -1:
        raise RuntimeError("Cannot find script block in HTML")

    samples_json_str = json.dumps(samples, ensure_ascii=False)

    new_script = f"""  <script>
  (function(){{
    'use strict';
    const $=id=>document.getElementById(id);
    const historyKey='tongji-ui-history-v1',prefsKey='tongji-ui-prefs-v1';
    function getStored(key,fallback){{try{{return JSON.parse(localStorage.getItem(key))??fallback}}catch{{return fallback}}}}
    function setStored(key,value){{try{{localStorage.setItem(key,JSON.stringify(value))}}catch{{}}}}

    const SAMPLES = {samples_json_str};

    const prefs = getStored(prefsKey, {{}});
    const defaultApi = (window.location.protocol === 'file:' ? 'http://localhost:8000/api/recognize' : '/api/recognize');
    const state = {{
      view: 'recognition',
      file: null,
      isSample: true,
      sampleIdx: 0,
      previewURL: null,
      recognizing: false,
      history: getStored(historyKey, []),
      demo: prefs.demo === true, // Default to FALSE: real AI inference
      api: typeof prefs.api === 'string' && prefs.api.trim() ? prefs.api.trim() : defaultApi,
      model: prefs.model || 'crc',
      sampleBlob: null
    }};

    let toastTimer;
    function toast(message,kind){{
      const el=$('toast');
      el.textContent=message;
      el.className='toast show'+(kind==='err'?' err':'');
      clearTimeout(toastTimer);
      toastTimer=setTimeout(()=>{{el.className='toast';}},3600);
    }}

    function iconUse(id,cls){{
      const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
      svg.setAttribute('class','ico'+(cls?' '+cls:''));
      const u=document.createElementNS('http://www.w3.org/2000/svg','use');
      u.setAttribute('href','#'+id);
      svg.appendChild(u);
      return svg;
    }}

    function setText(id,value){{
      const el=$(id);
      if(el)el.textContent=String(value);
    }}

    /* ---- Điều hướng ---- */
    function switchView(view){{
      if(!['recognition','history','about','settings'].includes(view))return;
      state.view=view;
      document.querySelectorAll('.subpage').forEach(el=>el.classList.toggle('active',el.dataset.page===view));
      document.querySelectorAll('.nav-item').forEach(el=>{{
        const on=el.dataset.view===view;
        el.classList.toggle('active',on);
        if(on)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');
      }});
      if(view==='history')renderHistory();
      if(view==='settings')syncSettings();
    }}
    document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.view)));
    document.querySelectorAll('[data-go]').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.go)));

    /* ---- Ảnh đầu vào ---- */
    function getPreview(){{return $('previewImg');}}
    function releaseObjectURL(){{
      if(state.previewURL){{URL.revokeObjectURL(state.previewURL);state.previewURL=null;}}
    }}

    /* ---- Khối kết quả ---- */
    function setResultState(has){{
      const card=$('resultCard');
      card.classList.toggle('is-empty',!has);
    }}
    function setArt(icon){{
      $('resultArt').replaceChildren(iconUse(icon));
    }}
    function setExtra(icon,text){{
      const el=$('resultExtra');
      el.replaceChildren(iconUse(icon),document.createTextNode(' ' + text));
    }}
    function resetResult(){{
      setText('identityValue','Chưa có kết quả');
      setText('identityLabel','Mã lòng bàn tay dự đoán');
      setText('resultDescription','Nhấn nút nhận dạng để xem kết quả.');
      setText('metricScore','—');
      setText('metricTime','—');
      $('resultBadge').className='chip chip-success';
      setText('resultBadge',state.demo?'MÔ PHỎNG':'AI THẬT');
      setArt('i-hand');
      setExtra('i-shield','Hỗ trợ 600 mã lòng bàn tay');
      const topWrap=$('top5Wrap');
      if(topWrap)topWrap.hidden=true;
      setResultState(false);
    }}

    /* ---- Áp dụng ảnh mẫu ---- */
    function applySample(idx){{
      releaseObjectURL();
      state.file=null;
      state.isSample=true;
      state.sampleIdx=(idx % SAMPLES.length);
      const cur=SAMPLES[state.sampleIdx];
      const img=getPreview();
      img.src=cur.data_url;
      setText('filename',cur.filename);
      setText('previewBadge','Ảnh mẫu Session 2 · Palm ID ' + String(cur.palm_id).padStart(4,'0'));
      setText('sizeBadge',cur.width + ' × ' + cur.height);
      document.querySelectorAll('.sample-chip').forEach((c,i)=>{{
        c.classList.toggle('active',i===state.sampleIdx);
      }});
      resetResult();
      updateRunState();
    }}

    async function loadFile(file){{
      if(!file)return;
      if(!['image/png','image/jpeg','image/webp','image/bmp'].includes(file.type)&&!/\\.(bmp|png|jpe?g|webp)$/i.test(file.name)){{
        toast('Chỉ hỗ trợ ảnh PNG, JPG, WebP hoặc BMP.','err');
        return;
      }}
      releaseObjectURL();
      state.file=file;
      state.isSample=false;
      state.previewURL=URL.createObjectURL(file);
      const img=getPreview();
      img.src=state.previewURL;
      img.onload=()=>{{
        setText('sizeBadge',img.naturalWidth+' × '+img.naturalHeight);
      }};
      setText('filename',file.name);
      setText('previewBadge','Ảnh ROI tải lên');
      document.querySelectorAll('.sample-chip').forEach(c=>c.classList.remove('active'));
      resetResult();
      updateRunState();
      toast('Đã nạp ảnh: ' + file.name);
    }}

    $('fileInput').addEventListener('change',e=>{{
      loadFile(e.target.files?.[0]);
      e.target.value='';
    }});
    $('uploadButton').addEventListener('click',()=>$('fileInput').click());
    $('sampleButton').addEventListener('click',()=>applySample(state.sampleIdx + 1));

    document.querySelectorAll('.sample-chip').forEach(btn=>{{
      btn.addEventListener('click',()=>{{
        const idx=parseInt(btn.dataset.sampleIdx||'0',10);
        applySample(idx);
      }});
    }});

    const dz=$('dropZone');
    dz.addEventListener('click',()=>$('fileInput').click());
    dz.addEventListener('keydown',e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();$('fileInput').click();}}}});
    ['dragenter','dragover'].forEach(evt=>dz.addEventListener(evt,e=>{{e.preventDefault();dz.classList.add('dragover');}}));
    ['dragleave','drop'].forEach(evt=>dz.addEventListener(evt,()=>dz.classList.remove('dragover')));
    dz.addEventListener('drop',e=>{{
      e.preventDefault();
      loadFile(e.dataTransfer?.files?.[0]);
    }});

    // Model Selector
    const modelSel = $('modelSelect');
    if(modelSel){{
      modelSel.value = state.model;
      modelSel.addEventListener('change',()=>{{
        state.model = modelSel.value;
        setStored(prefsKey,{{demo:state.demo,api:state.api,model:state.model}});
        resetResult();
        toast('Đã chọn thuật toán: ' + modelSel.options[modelSel.selectedIndex].text);
      }});
    }}

    /* ---- Cài đặt ---- */
    function syncSettings(){{
      setText('modeText',state.demo?'Chế độ minh họa':'Kết nối AI trực tiếp');
      $('modePill').classList.toggle('pill-demo',state.demo);
      $('demoSwitch').classList.toggle('on',state.demo);
      $('demoSwitch').setAttribute('aria-checked',String(state.demo));
      $('apiInput').value=state.api;
      setText('recognizeLabel',state.demo?'Thử nhận dạng demo':'Nhận dạng bằng AI');
      updateRunState();
    }}
    function switchDemo(){{
      state.demo=!state.demo;
      setStored(prefsKey,{{demo:state.demo,api:state.api,model:state.model}});
      syncSettings();
      resetResult();
      toast(state.demo?'Đã chuyển sang chế độ minh họa offline.':'Đã kết nối trực tiếp với mô hình AI.');
    }}
    $('demoSwitch').addEventListener('click',switchDemo);
    $('saveSettings').addEventListener('click',()=>{{
      state.api=$('apiInput').value.trim() || defaultApi;
      setStored(prefsKey,{{demo:state.demo,api:state.api,model:state.model}});
      toast('Đã lưu cấu hình endpoint nhận dạng.');
    }});

    /* ---- Lịch sử ---- */
    function saveHistory(record){{
      state.history.unshift({{
        id: String(record.id),
        model: String(record.model||'CRC').toUpperCase(),
        score: record.score??'—',
        latency: record.latency??'—',
        is_correct: record.is_correct,
        demo: !!record.demo,
        thumb: record.thumb || SAMPLES[state.sampleIdx % SAMPLES.length].data_url,
        when: new Date().toISOString()
      }});
      state.history=state.history.slice(0,60);
      setStored(historyKey,state.history);
      renderHistory();
    }}
    function newCell(row,value,cls){{
      const td=document.createElement('td');
      if(cls)td.className=cls;
      td.textContent=value;
      row.appendChild(td);
      return td;
    }}
    function timeString(iso){{
      const date=new Date(iso);
      return Number.isNaN(date.getTime())?'—':new Intl.DateTimeFormat('vi-VN',{{day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit'}}).format(date);
    }}
    function stateCell(row,is_correct,demo){{
      const td=document.createElement('td');
      const tag=document.createElement('span');
      if(demo){{
        tag.className='state-label sim';
        tag.textContent='Minh họa';
      }} else if(is_correct===true){{
        tag.className='state-label';
        tag.textContent='Chính xác';
      }} else if(is_correct===false){{
        tag.className='state-label sim';
        tag.textContent='Lệch nhãn';
      }} else {{
        tag.className='state-label';
        tag.textContent='AI Dự đoán';
      }}
      td.appendChild(tag);
      row.appendChild(td);
    }}

    function renderHistory(){{
      const all=$('historyRows'),recent=$('recentRows');
      all.replaceChildren();
      recent.replaceChildren();
      if(!state.history.length){{
        for(const target of [all,recent]){{
          const row=document.createElement('tr');
          const td=newCell(row,'Chưa có lượt nào. Chọn ảnh mẫu và nhấn Nhận dạng bằng AI.','empty-rows');
          td.colSpan=7;
          target.appendChild(row);
        }}
        return;
      }}
      state.history.forEach((record,i)=>{{
        // Wide table (History tab)
        const row=document.createElement('tr');
        newCell(row,String(i+1).padStart(2,'0'));
        newCell(row,timeString(record.when));
        newCell(row,'Palm ID ' + String(record.id).padStart(4,'0'),'identity-td');
        newCell(row,record.model||'CRC');
        newCell(row,String(record.score));
        newCell(row,String(record.latency));
        stateCell(row,record.is_correct,record.demo);
        all.appendChild(row);

        // Recent table (Recognition tab)
        if(i<4){{
          const mini=document.createElement('tr');
          const imgCell=document.createElement('td');
          const img=document.createElement('img');
          img.className='palm-tiny';
          img.alt='Ảnh';
          img.src=record.thumb || SAMPLES[0].data_url;
          imgCell.appendChild(img);
          mini.appendChild(imgCell);
          newCell(mini,timeString(record.when));
          newCell(mini,'Palm ID ' + String(record.id).padStart(4,'0'),'identity-td');
          newCell(mini,record.model||'CRC');
          newCell(mini,record.score + ' (' + record.latency + ')');
          recent.appendChild(mini);
        }}
      }});
    }}

    /* ---- Nhận dạng ---- */
    function renderTop5(top5){{
      const wrap=$('top5Wrap'),row=$('top5Row');
      if(!wrap||!row)return;
      row.replaceChildren();
      if(!Array.isArray(top5)||!top5.length){{wrap.hidden=true;return;}}
      top5.forEach((cand,idx)=>{{
        const item=document.createElement('div');
        item.className='top5-item' + (idx===0?' top1':'');
        const rank=document.createElement('div');
        rank.className='top5-rank';
        rank.textContent='#' + (idx+1);
        const id=document.createElement('div');
        id.className='top5-id';
        id.textContent=String(cand.palm_id).padStart(4,'0');
        const sc=document.createElement('div');
        sc.className='top5-score';
        sc.textContent=typeof cand.score==='number'?cand.score.toFixed(4):String(cand.score);
        item.appendChild(rank);
        item.appendChild(id);
        item.appendChild(sc);
        row.appendChild(item);
      }});
      wrap.hidden=false;
    }}

    function presentResult(res){{
      setText('identityValue','Palm ID ' + String(res.id).padStart(4,'0'));
      if(res.demo){{
        setText('resultDescription','Mã mẫu dùng để minh họa luồng nhận dạng.');
        $('resultBadge').className='chip chip-warn result-sim';
        setText('resultBadge','MÔ PHỎNG');
        setArt('i-tick');
        setExtra('i-info','Giao diện minh họa');
      }} else if(res.is_correct===true){{
        setText('resultDescription','Nhận dạng CHÍNH XÁC · Khớp với nhãn Ground Truth (Palm ID ' + String(res.id).padStart(4,'0') + ')');
        $('resultBadge').className='chip chip-success';
        setText('resultBadge','ĐÚNG 100%');
        setArt('i-tick');
        setExtra('i-check','Khớp nhãn Ground Truth');
      }} else if(res.is_correct===false){{
        setText('resultDescription','Dự đoán Palm ID ' + String(res.id).padStart(4,'0') + ' (Nhãn ground truth: Palm ID ' + String(res.true_id).padStart(4,'0') + ')');
        $('resultBadge').className='chip chip-warn';
        setText('resultBadge','KHÔNG TRÙNG');
        setArt('i-hand');
        setExtra('i-alert','Khác nhãn đối soát');
      }} else {{
        setText('resultDescription','Dự đoán thành công bởi mô hình ' + (res.model_name||state.model.toUpperCase()));
        $('resultBadge').className='chip chip-success';
        setText('resultBadge','AI THẬT');
        setArt('i-tick');
        setExtra('i-check','Phản hồi trực tiếp từ mô hình');
      }}

      setText('metricScore',res.score??'—');
      setText('metricName',res.kind??'Điểm so khớp');
      setText('metricTime',res.time??'—');
      setText('resultHint',res.demo?'KẾT QUẢ MÔ PHỎNG – không có AI suy luận.':'Suy luận trực tiếp từ checkpoint đã khóa của đề tài (Closed-set 600 lớp).');
      renderTop5(res.top5);
      setResultState(true);
    }}

    function updateRunState(){{
      $('recognizeButton').disabled=state.recognizing||(!state.file&&!state.isSample);
    }}

    function loading(on){{
      state.recognizing=on;
      updateRunState();
      $('scanLine').hidden=!on;
      setText('recognizeLabel',on?'Đang xử lý...':state.demo?'Thử nhận dạng demo':'Nhận dạng bằng AI');
    }}

    async function recognize(){{
      if(state.recognizing)return;

      // Offline demo fallback only if user explicitly checked demo mode in settings
      if(state.demo){{
        loading(true);
        await new Promise(r=>setTimeout(r,400));
        const sample=SAMPLES[state.sampleIdx % SAMPLES.length];
        presentResult({{
          id: sample.palm_id,
          score: '0.8954',
          kind: 'Phần dư tái tạo (Residual)',
          time: '15.5 ms',
          demo: true,
          top5: [
            {{palm_id: sample.palm_id, score: 0.8954}},
            {{palm_id: 163, score: 0.9365}},
            {{palm_id: 463, score: 0.9420}},
            {{palm_id: 374, score: 0.9427}},
            {{palm_id: 139, score: 0.9497}}
          ]
        }});
        saveHistory({{
          id: sample.palm_id,
          model: state.model.toUpperCase(),
          score: '0.8954',
          latency: '15.5 ms',
          is_correct: true,
          demo: true
        }});
        loading(false);
        toast('Đã hiển thị kết quả mô phỏng.');
        return;
      }}

      loading(true);
      try{{
        const fd=new FormData();
        if(state.isSample){{
          const sample=SAMPLES[state.sampleIdx % SAMPLES.length];
          const blob=await (await fetch(sample.data_url)).blob();
          fd.append('file',blob,sample.filename);
        }} else if(state.file){{
          fd.append('file',state.file,state.file.name);
        }} else {{
          throw new Error('Chưa chọn ảnh để nhận dạng');
        }}

        fd.append('model',state.model);

        const response=await fetch(state.api,{{
          method:'POST',
          body:fd,
          headers:{{'Accept':'application/json'}}
        }});

        if(!response.ok){{
          throw new Error('HTTP ' + response.status + ' (' + response.statusText + ')');
        }}

        const body=await response.json();
        const palm=body.palm_id??body.predicted_palm_id;
        if(palm===undefined||palm===null){{
          throw new Error('Phản hồi thiếu trường palm_id');
        }}

        const score=typeof body.score==='number'?body.score.toFixed(4):String(body.score||'—');
        const kind=body.score_kind_label || (typeof body.score_kind==='string'&&body.score_kind.includes('margin')?'Khoảng cách lề (-Margin)':'Phần dư tái tạo (Residual)');
        const elapsed=typeof body.latency_ms==='number'?body.latency_ms.toFixed(1)+' ms':'—';

        presentResult({{
          id: palm,
          score,
          kind,
          time: elapsed,
          demo: false,
          model_name: body.model_name,
          is_correct: body.is_correct,
          true_id: body.true_palm_id,
          top5: body.top5
        }});

        saveHistory({{
          id: palm,
          model: (body.model || state.model).toUpperCase(),
          score,
          latency: elapsed,
          is_correct: body.is_correct,
          demo: false,
          thumb: state.isSample ? SAMPLES[state.sampleIdx % SAMPLES.length].data_url : null
        }});

        toast('Đã nhận dạng thành công bằng mô hình ' + (body.model_name || state.model.toUpperCase()));
      }}catch(err){{
        toast('Không thể nhận dạng: ' + (err?.message||'Lỗi kết nối') + '. Hãy kiểm tra server app/server.py đang chạy tại ' + state.api,'err');
      }}finally{{
        loading(false);
      }}
    }}

    $('recognizeButton').addEventListener('click',recognize);

    let clearArm=0;
    const clearBtn=$('clearHistory'),clearHTML=clearBtn.innerHTML;
    clearBtn.addEventListener('click',()=>{{
      if(!state.history.length){{toast('Lịch sử đang trống.');return;}}
      if(!clearArm){{
        clearBtn.textContent='Bấm lần nữa để xóa';
        clearArm=setTimeout(()=>{{clearArm=0;clearBtn.innerHTML=clearHTML;}},3000);
        return;
      }}
      clearTimeout(clearArm);
      clearArm=0;
      clearBtn.innerHTML=clearHTML;
      state.history=[];
      setStored(historyKey,[]);
      renderHistory();
      toast('Đã xóa lịch sử nhận dạng.');
    }});

    /* ---- Giao diện sáng / tối ---- */
    function setTheme(theme){{
      document.documentElement.dataset.theme=theme;
      try{{localStorage.setItem('tongji-theme-v1',theme);}}catch(e){{}}
      $('themeButton').querySelector('use').setAttribute('href',theme==='dark'?'#i-sun':'#i-moon');
      const meta=$('themeColorMeta');
      if(meta)meta.setAttribute('content',theme==='dark'?'#0d1110':'#f4f4f1');
    }}
    setTheme((()=>{{
      try{{return localStorage.getItem('tongji-theme-v1')==='dark'?'dark':'light';}}catch{{return 'light';}}
    }})());
    $('themeButton').addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));

    /* Tự co giãn để toàn bộ giao diện nằm gọn trong một khung trình duyệt (không cần cuộn) */
    const DESIGN_W=1280,DESIGN_H=810;
    function fit(){{
      const w=window.innerWidth,h=window.innerHeight;
      let z=1;
      if(w>=900)z=Math.max(0.75,Math.min(w/DESIGN_W,h/DESIGN_H,2.4));
      document.documentElement.style.setProperty('--z',z.toFixed(3));
    }}
    fit();
    window.addEventListener('resize',fit);

    // Initial setup
    applySample(0);
    syncSettings();
    renderHistory();
    resetResult();
  }})();
  </script>"""

    html = html[:idx1] + new_script + html[idx2 + len(script_end) :]
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Successfully updated {HTML_PATH}, new size: {len(html):,} chars")


if __name__ == "__main__":
    update_ui()

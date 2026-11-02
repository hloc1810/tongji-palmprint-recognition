"""Update Tongji_Palmprint_UI.html:
1. Remove ALL hardcoded base64 images completely.
   All sample images and previews load dynamically via /api/sample-image/{id} and /api/samples.
2. Fix the layout strictly to 100vh with NO vertical or horizontal scrollbars (lock viewport).
"""

from __future__ import annotations

import re
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
HTML_PATH = WS / "Tongji_Palmprint_UI.html"


def update():
    html = HTML_PATH.read_text(encoding="utf-8")

    # 1. Replace hardcoded base64 in previewImg
    html = re.sub(
        r'<img id="previewImg"\s+src="data:image/png;base64,[^"]+"',
        '<img id="previewImg" src="/api/sample-image/1"',
        html,
    )

    # 2. Replace hardcoded base64 in roi-example
    html = re.sub(
        r'<div class="roi-example"><div class="preview-zone static"><img src="data:image/png;base64,[^"]+"',
        '<div class="roi-example"><div class="preview-zone static"><img src="/api/sample-image/1"',
        html,
    )

    # 3. Update CSS rules to lock the layout strictly to 100vh (NO scrolling)
    # Replace html/body zoom and overflow
    html = re.sub(
        r"html\{-webkit-text-size-adjust:100%;zoom:var\(--z\)\}",
        "html{width:100%;height:100vh;max-height:100vh;overflow:hidden;-webkit-text-size-adjust:100%;}",
        html,
    )
    html = re.sub(
        r"body\{margin:0;min-height:calc\(100vh / var\(--z\)\);",
        "body{margin:0;padding:0;width:100%;height:100vh;max-height:100vh;overflow:hidden;box-sizing:border-box;",
        html,
    )

    # App shell locked to 100vh
    html = re.sub(
        r"\.app\{display:grid;grid-template-columns:calc\(var\(--side-w\) \+ 16px\) minmax\(0,1fr\);min-height:calc\(100vh / var\(--z\)\)\}",
        ".app{display:grid;grid-template-columns:calc(var(--side-w) + 14px) minmax(0,1fr);width:100%;height:100vh;max-height:100vh;overflow:hidden;box-sizing:border-box;}",
        html,
    )

    # Side bar locked
    html = re.sub(
        r"\.side\{position:sticky;top:16px;height:calc\(100vh / var\(--z\) - 32px\);margin:16px 0 16px 16px;",
        ".side{position:sticky;top:8px;height:calc(100vh - 16px);margin:8px 0 8px 8px;padding:10px 6px;overflow:hidden;box-sizing:border-box;",
        html,
    )

    # Main locked
    html = re.sub(
        r"\.main\{min-width:0;display:flex;flex-direction:column\}",
        ".main{width:100%;height:100vh;max-height:100vh;overflow:hidden;display:flex;flex-direction:column;box-sizing:border-box;min-width:0;}",
        html,
    )

    # Topbar compact
    html = re.sub(
        r"\.topbar\{display:flex;align-items:center;justify-content:space-between;height:64px;",
        ".topbar{display:flex;align-items:center;justify-content:space-between;height:46px;flex-shrink:0;",
        html,
    )

    # Content locked
    html = re.sub(
        r"\.content\{width:100%;max-width:1200px;margin:0 auto;padding:4px 32px 20px;flex:1;display:flex;flex-direction:column\}",
        ".content{width:100%;max-width:1260px;margin:0 auto;padding:4px 20px 6px;flex:1;min-height:0;height:calc(100vh - 46px);max-height:calc(100vh - 46px);display:flex;flex-direction:column;overflow:hidden;box-sizing:border-box;}",
        html,
    )

    # Recognition subpage locked, auxiliary subpages scrollable internally
    subpage_old = '.subpage.active[data-page="recognition"]{display:flex;flex-direction:column;flex:1;min-height:0}'
    subpage_new = (
        '.subpage.active[data-page="recognition"]{display:flex;flex-direction:column;flex:1;min-height:0;height:100%;max-height:100%;overflow:hidden;}\n'
        '    .subpage.active:not([data-page="recognition"]){display:block;flex:1;min-height:0;height:100%;max-height:100%;overflow-y:auto;padding-right:6px;}'
    )
    if subpage_old in html:
        html = html.replace(subpage_old, subpage_new, 1)

    # Page head compact
    pagehead_old = ".page-head{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:16px}"
    pagehead_new = (
        ".page-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:6px;flex-shrink:0}\n"
        "    .page-head h2{font-size:18px;line-height:1.2}\n"
        "    .page-head p{font-size:12px;color:var(--ink-2);margin-top:1px}\n"
        "    .stat-strip .s{padding:3px 10px}\n"
        "    .stat-strip b{font-size:14px}\n"
        "    .stat-strip span{font-size:10.5px}\n"
        "    .stat-note{display:none}"
    )
    if pagehead_old in html:
        html = html.replace(pagehead_old, pagehead_new, 1)

    # Workspace & pane compact
    workspace_old = ".workspace{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));padding:0;flex:1;min-height:0}"
    workspace_new = (
        ".workspace{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));padding:0;flex:1;min-height:0;overflow:hidden}\n"
        "    .pane{min-width:0;padding:10px 16px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden}"
    )
    if workspace_old in html:
        html = html.replace(
            ".pane{min-width:0;padding:24px;display:flex;flex-direction:column}", "", 1
        )
        html = html.replace(workspace_old, workspace_new, 1)

    # History card compact
    hist_old = ".history-card{margin-top:20px}"
    hist_new = (
        ".history-card{margin-top:6px;padding:6px 14px;flex-shrink:0;max-height:102px;overflow:hidden}\n"
        "    .history-card .card-head{margin-bottom:2px}\n"
        "    .history-card table{font-size:11.5px}\n"
        "    .history-card td{padding:2px 6px}\n"
        "    .history-card .palm-tiny{width:22px;height:22px}\n"
        "    .footnote{display:flex;justify-content:space-between;margin-top:4px;padding-top:3px;font-size:10.5px;flex-shrink:0}"
    )
    if hist_old in html:
        html = html.replace(
            ".footnote{\n      display:flex;justify-content:space-between;gap:8px 24px;flex-wrap:wrap;\n      margin-top:20px;padding-top:12px;border-top:1px solid var(--line);\n      color:var(--ink-3);font-size:12px;\n    }",
            "",
            1,
        )
        html = html.replace(hist_old, hist_new, 1)

    # Compact input pane elements
    input_compact_css = """
    /* ---- Compact UI elements ---- */
    .input-content{display:grid;grid-template-columns:136px minmax(0,1fr);gap:12px;align-items:start;flex:1;min-height:0}
    .preview-zone{width:136px;height:136px}
    .preview-zone img{width:100%;height:100%;object-fit:contain}
    .preview-meta{margin-top:4px;font-size:11px}
    .step-caption{font-size:11px;font-weight:600;margin-bottom:2px}
    .sample-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin:3px 0 6px}
    .sample-chip{height:25px;font-size:11px;padding:0 4px}
    .model-group{margin:3px 0}
    .model-select{width:100%;height:32px;font-size:12px;padding:0 8px}
    .file-line{margin-top:2px;font-size:11px}
    .action-run{height:36px;font-size:12.5px;margin-top:4px;padding:0 14px}
    .helper-line{margin-top:2px;font-size:10.5px}

    .result-main{padding:8px 12px;gap:10px;margin-bottom:6px}
    .result-art{width:42px;height:42px}
    .result-art .ico{width:22px;height:22px}
    .identity{font-size:20px}
    .match-caption{font-size:11.5px;margin-top:2px}
    .ok-stat{margin-top:2px;font-size:11px}
    .result-info{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}
    .metric-box{padding:6px 10px}
    .metric-box .name{font-size:10.5px}
    .metric-box .value{font-size:14.5px;font-weight:700}
    .top5-wrap{margin-top:6px;padding-top:6px}
    .top5-title{font-size:10.5px;margin-bottom:4px}
    .top5-row{gap:4px}
    .top5-item{padding:3px 2px}
    .top5-rank{font-size:9.5px}
    .top5-id{font-size:11.5px;font-weight:700}
    .top5-score{font-size:9.5px}
    .result-hint{margin-top:4px;font-size:10.5px}
    """

    if ".input-content{display:grid" not in html:
        toast_marker = "    /* ==========================================================================\n       Toast"
        html = html.replace(toast_marker, input_compact_css + "\n" + toast_marker, 1)

    # 4. Update the entire script block with dynamic loading and NO base64
    script_start = "  <script>\n  (function(){\n    'use strict';"
    script_end = "  })();\n  </script>"

    idx1 = html.find(script_start)
    idx2 = html.find(script_end)
    if idx1 == -1 or idx2 == -1:
        raise RuntimeError("Cannot find script block in HTML")

    clean_script = """  <script>
  (function(){
    'use strict';
    const $=id=>document.getElementById(id);
    const historyKey='tongji-ui-history-v1',prefsKey='tongji-ui-prefs-v1';
    function getStored(key,fallback){try{return JSON.parse(localStorage.getItem(key))??fallback}catch{return fallback}}
    function setStored(key,value){try{localStorage.setItem(key,JSON.stringify(value))}catch{}}

    // Dynamic samples definition (URLs point to server API, zero hardcoded base64)
    let SAMPLES = [
      { id: 1, name: 'Mẫu 1 · ID 0001', filename: 'session2_00001.bmp', palm_id: 1, image_url: '/api/sample-image/1' },
      { id: 2, name: 'Mẫu 2 · ID 0042', filename: 'session2_00415.bmp', palm_id: 42, image_url: '/api/sample-image/2' },
      { id: 3, name: 'Mẫu 3 · ID 0128', filename: 'session2_01275.bmp', palm_id: 128, image_url: '/api/sample-image/3' },
      { id: 4, name: 'Mẫu 4 · ID 0300', filename: 'session2_02995.bmp', palm_id: 300, image_url: '/api/sample-image/4' },
      { id: 5, name: 'Mẫu 5 · ID 0427', filename: 'session2_04265.bmp', palm_id: 427, image_url: '/api/sample-image/5' },
      { id: 6, name: 'Mẫu 6 · ID 0599', filename: 'session2_05985.bmp', palm_id: 599, image_url: '/api/sample-image/6' }
    ];

    const prefs = getStored(prefsKey, {});
    const defaultApi = (window.location.protocol === 'file:' ? 'http://localhost:8000/api/recognize' : '/api/recognize');
    const apiBase = defaultApi.replace('/api/recognize', '');
    const state = {
      view: 'recognition',
      file: null,
      isSample: true,
      sampleIdx: 0,
      previewURL: null,
      recognizing: false,
      history: getStored(historyKey, []),
      demo: prefs.demo === true,
      api: typeof prefs.api === 'string' && prefs.api.trim() ? prefs.api.trim() : defaultApi,
      model: prefs.model || 'crc'
    };

    let toastTimer;
    function toast(message,kind){
      const el=$('toast');
      el.textContent=message;
      el.className='toast show'+(kind==='err'?' err':'');
      clearTimeout(toastTimer);
      toastTimer=setTimeout(()=>{el.className='toast';},3200);
    }

    function iconUse(id,cls){
      const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
      svg.setAttribute('class','ico'+(cls?' '+cls:''));
      const u=document.createElementNS('http://www.w3.org/2000/svg','use');
      u.setAttribute('href','#'+id);
      svg.appendChild(u);
      return svg;
    }

    function setText(id,value){
      const el=$(id);
      if(el)el.textContent=String(value);
    }

    /* ---- Điều hướng ---- */
    function switchView(view){
      if(!['recognition','history','about','settings'].includes(view))return;
      state.view=view;
      document.querySelectorAll('.subpage').forEach(el=>el.classList.toggle('active',el.dataset.page===view));
      document.querySelectorAll('.nav-item').forEach(el=>{
        const on=el.dataset.view===view;
        el.classList.toggle('active',on);
        if(on)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');
      });
      if(view==='history')renderHistory();
      if(view==='settings')syncSettings();
    }
    document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.view)));
    document.querySelectorAll('[data-go]').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.go)));

    /* ---- Ảnh đầu vào ---- */
    function getPreview(){return $('previewImg');}
    function releaseObjectURL(){
      if(state.previewURL){URL.revokeObjectURL(state.previewURL);state.previewURL=null;}
    }

    function resolveUrl(url){
      if(/^https?:\/\//i.test(url)) return url;
      return (window.location.protocol === 'file:' ? 'http://localhost:8000' : '') + url;
    }

    /* ---- Khối kết quả ---- */
    function setResultState(has){
      const card=$('resultCard');
      card.classList.toggle('is-empty',!has);
    }
    function setArt(icon){
      $('resultArt').replaceChildren(iconUse(icon));
    }
    function setExtra(icon,text){
      const el=$('resultExtra');
      el.replaceChildren(iconUse(icon),document.createTextNode(' ' + text));
    }
    function resetResult(){
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
    }

    /* ---- Áp dụng ảnh mẫu ---- */
    function applySample(idx){
      releaseObjectURL();
      state.file=null;
      state.isSample=true;
      state.sampleIdx=(idx % SAMPLES.length);
      const cur=SAMPLES[state.sampleIdx];
      const img=getPreview();
      img.src=resolveUrl(cur.image_url);
      setText('filename',cur.filename);
      setText('previewBadge','Ảnh mẫu Session 2 · Palm ID ' + String(cur.palm_id).padStart(4,'0'));
      setText('sizeBadge','128 × 128');
      document.querySelectorAll('.sample-chip').forEach((c,i)=>{
        c.classList.toggle('active',i===state.sampleIdx);
      });
      resetResult();
      updateRunState();
    }

    async function loadFile(file){
      if(!file)return;
      if(!['image/png','image/jpeg','image/webp','image/bmp'].includes(file.type)&&!/\\.(bmp|png|jpe?g|webp)$/i.test(file.name)){
        toast('Chỉ hỗ trợ ảnh PNG, JPG, WebP hoặc BMP.','err');
        return;
      }
      releaseObjectURL();
      state.file=file;
      state.isSample=false;
      state.previewURL=URL.createObjectURL(file);
      const img=getPreview();
      img.src=state.previewURL;
      img.onload=()=>{
        setText('sizeBadge',img.naturalWidth+' × '+img.naturalHeight);
      };
      setText('filename',file.name);
      setText('previewBadge','Ảnh ROI tải lên');
      document.querySelectorAll('.sample-chip').forEach(c=>c.classList.remove('active'));
      resetResult();
      updateRunState();
      toast('Đã nạp ảnh: ' + file.name);
    }

    $('fileInput').addEventListener('change',e=>{
      loadFile(e.target.files?.[0]);
      e.target.value='';
    });
    $('uploadButton').addEventListener('click',()=>$('fileInput').click());
    $('sampleButton').addEventListener('click',()=>applySample(state.sampleIdx + 1));

    document.querySelectorAll('.sample-chip').forEach(btn=>{
      btn.addEventListener('click',()=>{
        const idx=parseInt(btn.dataset.sampleIdx||'0',10);
        applySample(idx);
      });
    });

    const dz=$('dropZone');
    dz.addEventListener('click',()=>$('fileInput').click());
    dz.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('fileInput').click();}});
    ['dragenter','dragover'].forEach(evt=>dz.addEventListener(evt,e=>{e.preventDefault();dz.classList.add('dragover');}));
    ['dragleave','drop'].forEach(evt=>dz.addEventListener(evt,()=>dz.classList.remove('dragover')));
    dz.addEventListener('drop',e=>{
      e.preventDefault();
      loadFile(e.dataTransfer?.files?.[0]);
    });

    // Model Selector
    const modelSel = $('modelSelect');
    if(modelSel){
      modelSel.value = state.model;
      modelSel.addEventListener('change',()=>{
        state.model = modelSel.value;
        setStored(prefsKey,{demo:state.demo,api:state.api,model:state.model});
        resetResult();
        toast('Đã chọn thuật toán: ' + modelSel.options[modelSel.selectedIndex].text);
      });
    }

    /* ---- Cài đặt ---- */
    function syncSettings(){
      setText('modeText',state.demo?'Chế độ minh họa':'Kết nối AI trực tiếp');
      $('modePill').classList.toggle('pill-demo',state.demo);
      $('demoSwitch').classList.toggle('on',state.demo);
      $('demoSwitch').setAttribute('aria-checked',String(state.demo));
      $('apiInput').value=state.api;
      setText('recognizeLabel',state.demo?'Thử nhận dạng demo':'Nhận dạng bằng AI');
      updateRunState();
    }
    function switchDemo(){
      state.demo=!state.demo;
      setStored(prefsKey,{demo:state.demo,api:state.api,model:state.model});
      syncSettings();
      resetResult();
      toast(state.demo?'Đã chuyển sang chế độ minh họa offline.':'Đã kết nối trực tiếp với mô hình AI.');
    }
    $('demoSwitch').addEventListener('click',switchDemo);
    $('saveSettings').addEventListener('click',()=>{
      state.api=$('apiInput').value.trim() || defaultApi;
      setStored(prefsKey,{demo:state.demo,api:state.api,model:state.model});
      toast('Đã lưu cấu hình endpoint nhận dạng.');
    });

    /* ---- Lịch sử ---- */
    function saveHistory(record){
      state.history.unshift({
        id: String(record.id),
        model: String(record.model||'CRC').toUpperCase(),
        score: record.score??'—',
        latency: record.latency??'—',
        is_correct: record.is_correct,
        demo: !!record.demo,
        image_url: record.image_url || SAMPLES[state.sampleIdx % SAMPLES.length].image_url,
        when: new Date().toISOString()
      });
      state.history=state.history.slice(0,50);
      setStored(historyKey,state.history);
      renderHistory();
    }
    function newCell(row,value,cls){
      const td=document.createElement('td');
      if(cls)td.className=cls;
      td.textContent=value;
      row.appendChild(td);
      return td;
    }
    function timeString(iso){
      const date=new Date(iso);
      return Number.isNaN(date.getTime())?'—':new Intl.DateTimeFormat('vi-VN',{day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit'}).format(date);
    }
    function stateCell(row,is_correct,demo){
      const td=document.createElement('td');
      const tag=document.createElement('span');
      if(demo){
        tag.className='state-label sim';
        tag.textContent='Minh họa';
      } else if(is_correct===true){
        tag.className='state-label';
        tag.textContent='Chính xác';
      } else if(is_correct===false){
        tag.className='state-label sim';
        tag.textContent='Lệch nhãn';
      } else {
        tag.className='state-label';
        tag.textContent='AI Dự đoán';
      }
      td.appendChild(tag);
      row.appendChild(td);
    }

    function renderHistory(){
      const all=$('historyRows'),recent=$('recentRows');
      all.replaceChildren();
      recent.replaceChildren();
      if(!state.history.length){
        for(const target of [all,recent]){
          const row=document.createElement('tr');
          const td=newCell(row,'Chưa có lượt nào. Chọn ảnh mẫu và nhấn Nhận dạng bằng AI.','empty-rows');
          td.colSpan=7;
          target.appendChild(row);
        }
        return;
      }
      state.history.forEach((record,i)=>{
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
        if(i<3){
          const mini=document.createElement('tr');
          const imgCell=document.createElement('td');
          const img=document.createElement('img');
          img.className='palm-tiny';
          img.alt='Ảnh';
          img.src=resolveUrl(record.image_url || SAMPLES[0].image_url);
          imgCell.appendChild(img);
          mini.appendChild(imgCell);
          newCell(mini,timeString(record.when));
          newCell(mini,'Palm ID ' + String(record.id).padStart(4,'0'),'identity-td');
          newCell(mini,record.model||'CRC');
          newCell(mini,record.score + ' (' + record.latency + ')');
          recent.appendChild(mini);
        }
      });
    }

    /* ---- Nhận dạng ---- */
    function renderTop5(top5){
      const wrap=$('top5Wrap'),row=$('top5Row');
      if(!wrap||!row)return;
      row.replaceChildren();
      if(!Array.isArray(top5)||!top5.length){wrap.hidden=true;return;}
      top5.forEach((cand,idx)=>{
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
      });
      wrap.hidden=false;
    }

    function presentResult(res){
      setText('identityValue','Palm ID ' + String(res.id).padStart(4,'0'));
      if(res.demo){
        setText('resultDescription','Mã mẫu dùng để minh họa luồng nhận dạng.');
        $('resultBadge').className='chip chip-warn result-sim';
        setText('resultBadge','MÔ PHỎNG');
        setArt('i-tick');
        setExtra('i-info','Giao diện minh họa');
      } else if(res.is_correct===true){
        setText('resultDescription','Nhận dạng CHÍNH XÁC · Khớp Ground Truth (Palm ID ' + String(res.id).padStart(4,'0') + ')');
        $('resultBadge').className='chip chip-success';
        setText('resultBadge','ĐÚNG 100%');
        setArt('i-tick');
        setExtra('i-check','Khớp nhãn Ground Truth');
      } else if(res.is_correct===false){
        setText('resultDescription','Dự đoán Palm ID ' + String(res.id).padStart(4,'0') + ' (Nhãn ground truth: Palm ID ' + String(res.true_id).padStart(4,'0') + ')');
        $('resultBadge').className='chip chip-warn';
        setText('resultBadge','KHÔNG TRÙNG');
        setArt('i-hand');
        setExtra('i-alert','Khác nhãn đối soát');
      } else {
        setText('resultDescription','Dự đoán thành công bởi mô hình ' + (res.model_name||state.model.toUpperCase()));
        $('resultBadge').className='chip chip-success';
        setText('resultBadge','AI THẬT');
        setArt('i-tick');
        setExtra('i-check','Phản hồi trực tiếp từ mô hình');
      }

      setText('metricScore',res.score??'—');
      setText('metricName',res.kind??'Điểm so khớp');
      setText('metricTime',res.time??'—');
      setText('resultHint',res.demo?'KẾT QUẢ MÔ PHỎNG – không có AI suy luận.':'Suy luận trực tiếp từ checkpoint đã khóa của đề tài (Closed-set 600 lớp).');
      renderTop5(res.top5);
      setResultState(true);
    }

    function updateRunState(){
      $('recognizeButton').disabled=state.recognizing||(!state.file&&!state.isSample);
    }

    function loading(on){
      state.recognizing=on;
      updateRunState();
      $('scanLine').hidden=!on;
      setText('recognizeLabel',on?'Đang xử lý...':state.demo?'Thử nhận dạng demo':'Nhận dạng bằng AI');
    }

    async function recognize(){
      if(state.recognizing)return;

      if(state.demo){
        loading(true);
        await new Promise(r=>setTimeout(r,300));
        const sample=SAMPLES[state.sampleIdx % SAMPLES.length];
        presentResult({
          id: sample.palm_id,
          score: '0.8954',
          kind: 'Phần dư tái tạo (Residual)',
          time: '15.5 ms',
          demo: true,
          top5: [
            {palm_id: sample.palm_id, score: 0.8954},
            {palm_id: 163, score: 0.9365},
            {palm_id: 463, score: 0.9420},
            {palm_id: 374, score: 0.9427},
            {palm_id: 139, score: 0.9497}
          ]
        });
        saveHistory({
          id: sample.palm_id,
          model: state.model.toUpperCase(),
          score: '0.8954',
          latency: '15.5 ms',
          is_correct: true,
          demo: true,
          image_url: sample.image_url
        });
        loading(false);
        toast('Đã hiển thị kết quả mô phỏng.');
        return;
      }

      loading(true);
      try{
        const fd=new FormData();
        let currentThumbUrl = null;
        if(state.isSample){
          const sample=SAMPLES[state.sampleIdx % SAMPLES.length];
          const fullImgUrl=resolveUrl(sample.image_url);
          const response=await fetch(fullImgUrl);
          if(!response.ok) throw new Error('Không thể tải ảnh mẫu từ ' + fullImgUrl);
          const blob=await response.blob();
          fd.append('file',blob,sample.filename);
          currentThumbUrl = sample.image_url;
        } else if(state.file){
          fd.append('file',state.file,state.file.name);
          currentThumbUrl = state.previewURL;
        } else {
          throw new Error('Chưa chọn ảnh để nhận dạng');
        }

        fd.append('model',state.model);

        const response=await fetch(resolveUrl(state.api),{
          method:'POST',
          body:fd,
          headers:{'Accept':'application/json'}
        });

        if(!response.ok){
          throw new Error('HTTP ' + response.status + ' (' + response.statusText + ')');
        }

        const body=await response.json();
        const palm=body.palm_id??body.predicted_palm_id;
        if(palm===undefined||palm===null){
          throw new Error('Phản hồi thiếu trường palm_id');
        }

        const score=typeof body.score==='number'?body.score.toFixed(4):String(body.score||'—');
        const kind=body.score_kind_label || (typeof body.score_kind==='string'&&body.score_kind.includes('margin')?'Khoảng cách lề (-Margin)':'Phần dư tái tạo (Residual)');
        const elapsed=typeof body.latency_ms==='number'?body.latency_ms.toFixed(1)+' ms':'—';

        presentResult({
          id: palm,
          score,
          kind,
          time: elapsed,
          demo: false,
          model_name: body.model_name,
          is_correct: body.is_correct,
          true_id: body.true_palm_id,
          top5: body.top5
        });

        saveHistory({
          id: palm,
          model: (body.model || state.model).toUpperCase(),
          score,
          latency: elapsed,
          is_correct: body.is_correct,
          demo: false,
          image_url: currentThumbUrl
        });

        toast('Đã nhận dạng thành công bằng mô hình ' + (body.model_name || state.model.toUpperCase()));
      }catch(err){
        toast('Không thể nhận dạng: ' + (err?.message||'Lỗi kết nối') + '. Hãy kiểm tra server app/server.py đang chạy tại ' + state.api,'err');
      }finally{
        loading(false);
      }
    }

    $('recognizeButton').addEventListener('click',recognize);

    let clearArm=0;
    const clearBtn=$('clearHistory'),clearHTML=clearBtn.innerHTML;
    clearBtn.addEventListener('click',()=>{
      if(!state.history.length){toast('Lịch sử đang trống.');return;}
      if(!clearArm){
        clearBtn.textContent='Bấm lần nữa để xóa';
        clearArm=setTimeout(()=>{clearArm=0;clearBtn.innerHTML=clearHTML;},3000);
        return;
      }
      clearTimeout(clearArm);
      clearArm=0;
      clearBtn.innerHTML=clearHTML;
      state.history=[];
      setStored(historyKey,[]);
      renderHistory();
      toast('Đã xóa lịch sử nhận dạng.');
    });

    /* ---- Giao diện sáng / tối ---- */
    function setTheme(theme){
      document.documentElement.dataset.theme=theme;
      try{localStorage.setItem('tongji-theme-v1',theme);}catch(e){}
      $('themeButton').querySelector('use').setAttribute('href',theme==='dark'?'#i-sun':'#i-moon');
      const meta=$('themeColorMeta');
      if(meta)meta.setAttribute('content',theme==='dark'?'#0d1110':'#f4f4f1');
    }
    setTheme((()=>{
      try{return localStorage.getItem('tongji-theme-v1')==='dark'?'dark':'light';}catch{return 'light';}
    })());
    $('themeButton').addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));

    // Dynamic samples fetch from API
    async function loadSamplesList(){
      try{
        const r=await fetch(resolveUrl('/api/samples'));
        if(r.ok){
          const list=await r.json();
          if(Array.isArray(list)&&list.length>0){
            SAMPLES=list;
          }
        }
      }catch(e){}
    }

    // Initial setup
    loadSamplesList();
    applySample(0);
    syncSettings();
    renderHistory();
    resetResult();
  })();
  </script>"""

    html = html[:idx1] + clean_script + html[idx2 + len(script_end) :]
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Successfully updated {HTML_PATH}, clean size: {len(html):,} chars")


if __name__ == "__main__":
    update()

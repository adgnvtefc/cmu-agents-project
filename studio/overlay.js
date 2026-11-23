// Studio overlay: floats over the workspace. Talks to Python through the __studio binding.
(() => {
  if (window.top !== window || window.__studioInstalled) return;
  window.__studioInstalled = true;
  const call = (op, data = {}) => window.__studio({ op, ...data });

  const host = document.createElement('div');
  host.id = '__studio';
  const root = host.attachShadow({ mode: 'open' });
  root.innerHTML = `<style>
    .box { position: fixed; right: 16px; bottom: 16px; z-index: 2147483647; width: 260px; padding: 14px;
      background: #1d2430; color: #fff; border-radius: 12px; box-shadow: 0 8px 28px rgba(0,0,0,.28);
      font: 13px/1.4 -apple-system, "Helvetica Neue", Arial, sans-serif; }
    button { font: inherit; font-weight: 600; border: 0; border-radius: 8px; padding: 8px 12px; cursor: pointer;
      background: #2f6fde; color: #fff; }
    button:disabled { opacity: .4; cursor: default; }
    button.rec { background: #e03b3b; } button.alt { background: #3a4456; }
    input { width: 100%; box-sizing: border-box; font: inherit; padding: 7px 9px; border-radius: 7px;
      border: 1px solid #3a4456; background: #273040; color: #fff; margin-bottom: 8px; }
    .name { font-weight: 700; font-size: 14px; margin-bottom: 8px; }
    .bar { display: flex; gap: 4px; align-items: center; margin-bottom: 10px; }
    .bar i { flex: 1; height: 8px; border-radius: 4px; background: #3a4456; }
    .bar i.on { background: #3fb871; }
    .bar span { margin-left: 6px; font-weight: 700; }
    .row { display: flex; gap: 6px; } .row button { flex: 1; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #e03b3b; margin-right: 6px;
      animation: p 1.2s infinite; } @keyframes p { 50% { opacity: .2; } }
    .msg { margin-bottom: 8px; color: #c6ccd6; } .err { color: #ff9c9c; margin-top: 6px; }
    .prog { height: 6px; background: #3a4456; border-radius: 3px; overflow: hidden; margin-top: 6px; }
    .prog div { height: 100%; background: #2f6fde; }
  </style><div class="box"></div>`;
  const box = root.querySelector('.box');
  let S = null, typed = '';

  const bar = (n, goal) => `<div class="bar">${Array.from({ length: goal }, (_, i) => `<i class="${i < n ? 'on' : ''}"></i>`).join('')}<span>${n}/${goal}</span></div>`;
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function render() {
    if (!S) return;
    let h = '';
    if (S.phase === 'none') {
      h = `<button data-op="new">Create skill</button>`;
    } else if (S.phase === 'naming') {
      h = `<input id="name" placeholder="Skill name" autofocus><div class="row"><button data-op="create">Create</button><button class="alt" data-op="cancel">Cancel</button></div>`;
    } else {
      h = `<div class="name">${esc(S.skill)}</div>${bar(S.demos, S.goal)}`;
      if (S.phase === 'idle') {
        h += `<input id="instr" placeholder="What will you do in this demo?" value="${esc(typed)}"><button data-op="record" ${typed.trim() ? '' : 'disabled'}>Record</button>`;
      } else if (S.phase === 'recording' || S.phase === 'stopped') {
        h += S.phase === 'recording' ? `<div class="msg"><span class="dot"></span>Recording: ${esc(S.instruction)}</div>`
                                     : `<div class="msg">Stopped (${S.steps} steps): ${esc(S.instruction)}</div>`;
        h += `<div class="row"><button class="rec" data-op="stop" ${S.phase === 'recording' ? '' : 'disabled'}>Stop</button>
              <button data-op="submit">Submit</button><button class="alt" data-op="redo">Redo</button></div>`;
      } else if (S.phase === 'training') {
        const pct = S.train.total ? Math.round(100 * S.train.step / S.train.total) : 0;
        h += `<div class="msg">Training${S.train.total ? ` · ${pct}%` : ' · loading model…'}</div><div class="prog"><div style="width:${pct}%"></div></div>`;
      } else if (S.phase === 'ready') {
        h += `<div class="msg">✓ Skill ready</div><button class="alt" data-op="new">Create skill</button>`;
      }
    }
    if (S.error) h += `<div class="err">${esc(S.error)}</div>`;
    box.innerHTML = h;
    const name = root.getElementById('name');
    if (name) { name.focus(); name.onkeydown = (e) => { if (e.key === 'Enter') act('create'); }; }
    const instr = root.getElementById('instr');
    if (instr) instr.oninput = () => { typed = instr.value; root.querySelector('[data-op=record]').disabled = !typed.trim(); };
  }

  async function act(op) {
    const data = {};
    if (op === 'new') { S = { ...S, phase: 'naming', error: null }; return render(); }
    if (op === 'cancel') { S = await call('state'); return render(); }
    if (op === 'create') data.name = (root.getElementById('name') || {}).value || '';
    if (op === 'record') data.instruction = typed.trim();
    box.querySelectorAll('button').forEach((b) => (b.disabled = true));  // busy until Python answers
    S = await call(op, data);
    if (op === 'submit') typed = '';
    render();
  }
  box.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b && !b.disabled) act(b.dataset.op); });

  async function poll() {
    try {
      if (!S || S.phase !== 'naming') {
        const next = await call('state');
        const focused = root.activeElement && root.activeElement.tagName === 'INPUT';
        if (!S || JSON.stringify(next) !== JSON.stringify(S)) { S = next; if (!focused || S.phase !== 'idle') render(); }
      }
    } catch (e) {}
    setTimeout(poll, S && (S.phase === 'training' || S.phase === 'recording') ? 1000 : 2500);
  }
  const mount = () => { document.body.appendChild(host); poll(); };
  document.body ? mount() : document.addEventListener('DOMContentLoaded', mount);
})();

// Injected into every page. Reports user input to the recorder via the __rec binding.
// Capture phase so we see events before page handlers / navigation.
(() => {
  if (window.__recInstalled) return;
  window.__recInstalled = true;
  const send = (payload) => { try { window.__rec(payload); } catch (e) {} };

  const INTERACTIVE = 'a,button,input,textarea,select,label,[role=button],[data-testid]';
  const describe = (el) => {
    if (!el || !el.getBoundingClientRect) return null;
    const t = el.closest(INTERACTIVE) || el;
    const r = t.getBoundingClientRect();
    return {
      tag: t.tagName.toLowerCase(),
      testid: t.getAttribute('data-testid'),
      text: (t.innerText || t.value || '').trim().slice(0, 80),
      name: t.getAttribute('name'),
      type: t.getAttribute('type'),
      placeholder: t.getAttribute('placeholder'),
      href: t.getAttribute('href'),
      bbox: [r.left, r.top, r.width, r.height],
    };
  };
  const editable = (el) => el && (el.tagName === 'TEXTAREA' ||
    (el.tagName === 'INPUT' && !['radio', 'checkbox', 'submit', 'button', 'hidden'].includes(el.type)));

  let lastLabel = null, lastLabelT = 0;
  document.addEventListener('click', (e) => {
    if (e.detail === 0) return;  // synthetic (keyboard activation)
    // Clicking a <label> also dispatches a click on its control: drop that duplicate.
    if (lastLabel && lastLabel.control === e.target && performance.now() - lastLabelT < 200) {
      lastLabel = null;
      return;
    }
    const lbl = e.target.closest && e.target.closest('label');
    if (lbl && lbl.control && lbl.control !== e.target) { lastLabel = lbl; lastLabelT = performance.now(); }
    send({kind: 'click', x: e.clientX, y: e.clientY, button: e.button, detail: e.detail,
          target: describe(e.target), url: location.href});
  }, true);

  document.addEventListener('contextmenu', (e) => {
    send({kind: 'click', x: e.clientX, y: e.clientY, button: 2, detail: 1,
          target: describe(e.target), url: location.href});
  }, true);

  document.addEventListener('keydown', (e) => {
    const el = e.target;
    send({kind: 'keydown', key: e.key, meta: e.metaKey, ctrl: e.ctrlKey, alt: e.altKey, shift: e.shiftKey,
          editable: editable(el), value: editable(el) ? el.value : null,
          sel: editable(el) ? [el.selectionStart, el.selectionEnd] : null,
          target: describe(el), url: location.href});
  }, true);

  document.addEventListener('input', (e) => {
    const el = e.target;
    if (!editable(el)) return;
    send({kind: 'input', inputType: e.inputType, value: el.value, target: describe(el), url: location.href});
  }, true);

  document.addEventListener('wheel', (e) => {
    send({kind: 'wheel', x: e.clientX, y: e.clientY, dx: e.deltaX, dy: e.deltaY,
          target: describe(e.target), url: location.href});
  }, {capture: true, passive: true});
})();

// Injected into every page. Reports user input to the recorder via the __rec binding.
// Capture phase so we see events before page handlers / navigation.
(() => {
  if (window.__recInstalled) return;
  window.__recInstalled = true;
  // ts: wall-clock time of the event itself (the binding call may be delivered late under load)
  const send = (payload) => { try { window.__rec({...payload, ts: Date.now()}); } catch (e) {} };
  // Input on the Studio overlay is not part of the demo.
  const fromStudio = (e) => (e.composedPath ? e.composedPath() : []).some((n) => n && n.id === '__studio');

  const INTERACTIVE = 'a,button,input,textarea,select,label,[role=button],[data-testid]';
  const describe = (el) => {
    if (!el || !el.getBoundingClientRect) return null;
    const t = el.closest(INTERACTIVE) || el;
    const r = t.getBoundingClientRect();
    return {
      tag: t.tagName.toLowerCase(),
      id: identity(t),
      label: nameOf(t),
      testid: t.getAttribute('data-testid'),
      text: (t.innerText || t.value || '').trim().slice(0, 80),
      name: t.getAttribute('name'),
      type: t.getAttribute('type'),
      placeholder: t.getAttribute('placeholder'),
      href: t.getAttribute('href'),
      bbox: [r.left, r.top, r.width, r.height],
    };
  };
  // Human-readable name of an element (label text for fields, visible text for buttons/links).
  const nameOf = (t) => {
    if (!t) return '';
    if (t.labels && t.labels.length) {
      const lbl = t.labels[0].cloneNode(true);
      lbl.querySelectorAll('input,textarea,select').forEach((x) => x.remove());
      return lbl.innerText.trim().slice(0, 60);
    }
    if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA') {  // never name a field by its value
      return (t.getAttribute('aria-label') || t.getAttribute('placeholder') || t.getAttribute('name') || '').trim().slice(0, 60);
    }
    return (t.innerText || t.getAttribute('aria-label') || t.value || '').trim().slice(0, 60);
  };

  // Identity of what a click at an element would activate: a <label> activates its control.
  const identity = (t) => {
    let el = t;
    if (el.tagName === 'LABEL' && el.control) el = el.control;
    return el.getAttribute('data-testid') || `${el.tagName.toLowerCase()}:${nameOf(el)}`;
  };

  // Snapshot of every visible interactive element: used to score predicted clicks offline
  // (a predicted point is correct if it would activate the same element as the gold click).
  window.__elementMap = () => {
    const out = [];
    document.querySelectorAll(INTERACTIVE).forEach((t) => {
      const r = t.getBoundingClientRect();
      if (r.width < 1 || r.height < 1 || r.bottom < 0 || r.top > innerHeight) return;
      const cs = getComputedStyle(t);
      if (cs.visibility === 'hidden' || cs.display === 'none') return;
      out.push({id: identity(t), name: nameOf(t), tag: t.tagName.toLowerCase(),
                bbox: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]});
    });
    return out;
  };

  const editable = (el) => el && (el.tagName === 'TEXTAREA' ||
    (el.tagName === 'INPUT' && !['radio', 'checkbox', 'submit', 'button', 'hidden'].includes(el.type)));

  let lastLabel = null, lastLabelT = 0;
  document.addEventListener('click', (e) => {
    if (fromStudio(e)) return;
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
    if (fromStudio(e)) return;
    send({kind: 'click', x: e.clientX, y: e.clientY, button: 2, detail: 1,
          target: describe(e.target), url: location.href});
  }, true);

  document.addEventListener('keydown', (e) => {
    if (fromStudio(e)) return;
    const el = e.target;
    send({kind: 'keydown', key: e.key, meta: e.metaKey, ctrl: e.ctrlKey, alt: e.altKey, shift: e.shiftKey,
          editable: editable(el), value: editable(el) ? el.value : null,
          sel: editable(el) ? [el.selectionStart, el.selectionEnd] : null,
          target: describe(el), url: location.href});
  }, true);

  document.addEventListener('input', (e) => {
    if (fromStudio(e)) return;
    const el = e.target;
    if (!editable(el)) return;
    send({kind: 'input', inputType: e.inputType, value: el.value, target: describe(el), url: location.href});
  }, true);

  document.addEventListener('wheel', (e) => {
    if (fromStudio(e)) return;
    send({kind: 'wheel', x: e.clientX, y: e.clientY, dx: e.deltaX, dy: e.deltaY,
          target: describe(e.target), url: location.href});
  }, {capture: true, passive: true});
})();

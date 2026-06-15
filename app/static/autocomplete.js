/**
 * Drug autocomplete widget
 * Usage: initAutocomplete('input-id')
 */
function initAutocomplete(inputId) {
  const input = document.getElementById(inputId);
  if (!input) return;
  const wrap = input.parentElement;
  wrap.style.position = 'relative';
  const dropdown = document.createElement('div');
  dropdown.style.cssText = 'position:absolute;top:100%;left:0;right:0;z-index:9999;background:#1e293b;border:1px solid #38bdf8;border-top:none;border-radius:0 0 10px 10px;max-height:240px;overflow-y:auto;display:none;box-shadow:0 8px 24px rgba(0,0,0,0.4);';
  wrap.appendChild(dropdown);
  let timer = null, selectedIdx = -1, items = [];

  input.addEventListener('input', function() {
    clearTimeout(timer);
    const q = this.value.trim();
    if (q.length < 2) { dropdown.style.display = 'none'; return; }
    timer = setTimeout(() => fetchSuggestions(q), 200);
  });

  input.addEventListener('keydown', function(e) {
    if (dropdown.style.display === 'none') return;
    if (e.key === 'ArrowDown') { e.preventDefault(); moveSelection(1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); moveSelection(-1); }
    else if (e.key === 'Enter' && selectedIdx >= 0) { e.preventDefault(); selectItem(items[selectedIdx]); }
    else if (e.key === 'Escape') { dropdown.style.display = 'none'; }
  });

  document.addEventListener('click', function(e) {
    if (!wrap.contains(e.target)) dropdown.style.display = 'none';
  });

  async function fetchSuggestions(q) {
    try {
      const res = await fetch('/api/autocomplete?q=' + encodeURIComponent(q.toUpperCase()));
      items = await res.json();
      renderDropdown(q);
    } catch(e) { dropdown.style.display = 'none'; }
  }

  function renderDropdown(q) {
    if (items.length === 0) { dropdown.style.display = 'none'; return; }
    selectedIdx = -1;
    dropdown.innerHTML = items.map((drug, i) => {
      const hl = drug.replace(new RegExp('^(' + q.toUpperCase() + ')', 'i'), '<span style="color:#38bdf8;font-weight:bold;">$1</span>');
      return '<div data-idx="' + i + '" style="padding:10px 16px;cursor:pointer;font-size:0.9rem;color:#e2e8f0;border-bottom:1px solid #334155;" onmouseover="this.style.background=\'#334155\'" onmouseout="this.style.background=\'\'" onclick="document.getElementById(\'' + inputId + '\').value=\'' + drug + '\';this.parentElement.style.display=\'none\'">' + hl + '</div>';
    }).join('');
    dropdown.style.display = 'block';
  }

  function moveSelection(dir) {
    const divs = dropdown.querySelectorAll('div');
    if (divs.length === 0) return;
    if (selectedIdx >= 0) divs[selectedIdx].style.background = '';
    selectedIdx = Math.max(0, Math.min(items.length - 1, selectedIdx + dir));
    divs[selectedIdx].style.background = '#334155';
    input.value = items[selectedIdx];
  }

  function selectItem(drug) {
    input.value = drug;
    dropdown.style.display = 'none';
    selectedIdx = -1;
  }
}

function initReactionAutocomplete(inputId) {
  const input = document.getElementById(inputId);
  if (!input) return;
  const wrap = input.parentElement;
  wrap.style.position = 'relative';
  const dropdown = document.createElement('div');
  dropdown.style.cssText = 'position:absolute;top:100%;left:0;right:0;z-index:9999;background:#1e293b;border:1px solid #a855f7;border-top:none;border-radius:0 0 10px 10px;max-height:240px;overflow-y:auto;display:none;box-shadow:0 8px 24px rgba(0,0,0,0.4);';
  wrap.appendChild(dropdown);
  let timer = null, selectedIdx = -1, items = [];

  input.addEventListener('input', function() {
    clearTimeout(timer);
    const q = this.value.trim();
    if (q.length < 2) { dropdown.style.display = 'none'; return; }
    timer = setTimeout(() => fetchSuggestions(q), 200);
  });

  input.addEventListener('keydown', function(e) {
    if (dropdown.style.display === 'none') return;
    if (e.key === 'ArrowDown') { e.preventDefault(); moveSelection(1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); moveSelection(-1); }
    else if (e.key === 'Enter' && selectedIdx >= 0) { e.preventDefault(); selectItem(items[selectedIdx]); }
    else if (e.key === 'Escape') { dropdown.style.display = 'none'; }
  });

  document.addEventListener('click', function(e) {
    if (!wrap.contains(e.target)) dropdown.style.display = 'none';
  });

  async function fetchSuggestions(q) {
    try {
      const res = await fetch('/api/autocomplete/reaction?q=' + encodeURIComponent(q.toUpperCase()));
      items = await res.json();
      renderDropdown(q);
    } catch(e) { dropdown.style.display = 'none'; }
  }

  function renderDropdown(q) {
    if (items.length === 0) { dropdown.style.display = 'none'; return; }
    selectedIdx = -1;
    dropdown.innerHTML = items.map((reac, i) => {
      const hl = reac.replace(new RegExp('^(' + q.toUpperCase() + ')', 'i'), '<span style="color:#a855f7;font-weight:bold;">$1</span>');
      return '<div data-idx="' + i + '" style="padding:10px 16px;cursor:pointer;font-size:0.9rem;color:#e2e8f0;border-bottom:1px solid #334155;" onmouseover="this.style.background=\'#334155\'" onmouseout="this.style.background=\'\'" onclick="document.getElementById(\'' + inputId + '\').value=\'' + reac + '\';this.parentElement.style.display=\'none\'">' + hl + '</div>';
    }).join('');
    dropdown.style.display = 'block';
  }

  function moveSelection(dir) {
    const divs = dropdown.querySelectorAll('div');
    if (divs.length === 0) return;
    if (selectedIdx >= 0) divs[selectedIdx].style.background = '';
    selectedIdx = Math.max(0, Math.min(items.length - 1, selectedIdx + dir));
    divs[selectedIdx].style.background = '#334155';
    input.value = items[selectedIdx];
  }

  function selectItem(reac) {
    input.value = reac;
    dropdown.style.display = 'none';
    selectedIdx = -1;
  }
}

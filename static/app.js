document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('summarize-form');
  const dialogueInput = document.getElementById('dialogue');
  const summarizeBtn = document.getElementById('summarize-btn');
  const btnLabel = summarizeBtn.querySelector('.btn-label');
  const summaryContainer = document.getElementById('summary-container');
  const charCounter = document.getElementById('char-counter');
  const clearBtn = document.getElementById('clear-btn');
  const copyBtn = document.getElementById('copy-btn');
  const copyText = copyBtn.querySelector('.copy-text');

  function updateCharCount() {
    const len = dialogueInput.value.length;
    charCounter.textContent = `${len.toLocaleString()} characters`;
  }

  function syncCopyState() {
    const summaryElem = document.getElementById('summary-text');
    copyBtn.disabled = !(summaryElem && summaryElem.textContent.trim().length > 0);
  }

  dialogueInput.addEventListener('input', updateCharCount);
  updateCharCount();
  syncCopyState();

  dialogueInput.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!summarizeBtn.disabled) {
        form.requestSubmit();
      }
    }
  });

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      dialogueInput.value = '';
      updateCharCount();

      summaryContainer.innerHTML = `
        <div class="empty-state" id="placeholder-text">
          <p>Your generated summary will appear here.</p>
        </div>
      `;

      copyBtn.disabled = true;
      copyBtn.classList.remove('copied');
      copyText.textContent = 'Copy';
      dialogueInput.focus();
    });
  }

  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      const summaryElem = document.getElementById('summary-text');
      if (!summaryElem || !summaryElem.textContent.trim()) return;

      const text = summaryElem.textContent.trim();
      let ok = false;

      if (navigator.clipboard && window.isSecureContext) {
        try {
          await navigator.clipboard.writeText(text);
          ok = true;
        } catch {
          // fallback below
        }
      }

      if (!ok) {
        try {
          const ta = document.createElement('textarea');
          ta.value = text;
          ta.style.position = 'fixed';
          ta.style.left = '-9999px';
          document.body.appendChild(ta);
          ta.select();
          ok = document.execCommand('copy');
          document.body.removeChild(ta);
        } catch (err) {
          console.error(err);
        }
      }

      if (ok) {
        const origText = copyText.textContent;
        copyBtn.classList.add('copied');
        copyText.textContent = 'Copied!';

        setTimeout(() => {
          copyBtn.classList.remove('copied');
          copyText.textContent = origText;
        }, 2000);
      }
    });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const text = dialogueInput.value.trim();
    if (!text) {
      dialogueInput.classList.add('shake');
      dialogueInput.focus();
      setTimeout(() => dialogueInput.classList.remove('shake'), 350);
      return;
    }

    summarizeBtn.disabled = true;
    summarizeBtn.classList.add('loading');
    btnLabel.textContent = 'Summarizing...';
    copyBtn.disabled = true;

    summaryContainer.innerHTML = `
      <div class="skeleton-loader" aria-label="Loading summary...">
        <div class="skeleton-line"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line"></div>
      </div>
    `;

    try {
      const formData = new FormData(form);
      const response = await fetch('/summarize', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const contentType = response.headers.get('content-type') || '';
      let result = '';

      if (contentType.includes('application/json')) {
        const data = await response.json();
        result = data.summary || data.result || 'No summary returned.';
      } else {
        const html = await response.text();
        const doc = new DOMParser().parseFromString(html, 'text/html');
        const parsed = doc.querySelector('#summary-text') || doc.querySelector('.summary-box');
        result = parsed ? parsed.textContent.trim() : 'Summary generated.';
      }

      summaryContainer.innerHTML = `
        <div class="summary-box" id="summary-text">${escapeHtml(result)}</div>
      `;
      copyBtn.disabled = false;

    } catch (err) {
      console.error(err);
      summaryContainer.innerHTML = `
        <div class="empty-state" style="color: #ef4444;">
          <p>Failed to generate summary. Please verify the server is running.</p>
        </div>
      `;
      copyBtn.disabled = true;
    } finally {
      summarizeBtn.disabled = false;
      summarizeBtn.classList.remove('loading');
      btnLabel.textContent = 'Summarize';
    }
  });

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
});

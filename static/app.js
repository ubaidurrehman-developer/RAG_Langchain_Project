/**
 * Ubaid's Retrieval System - Frontend Client Console
 * Clean, high-precision document vector ingestion and retrieval interface.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM References
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  
  const uploadSection = document.getElementById('upload-section');
  const uploadProgressCard = document.getElementById('upload-progress-card');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const progressStepText = document.getElementById('progress-step-text');
  const progressPercent = document.getElementById('progress-percent');
  const progressSubdetail = document.getElementById('progress-subdetail');
  
  const chatSection = document.getElementById('chat-section');
  const activeDocFilename = document.getElementById('active-doc-filename');
  const metricChunks = document.getElementById('metric-chunks');
  const metricPages = document.getElementById('metric-pages');
  const metricChars = document.getElementById('metric-chars');
  const changeDocBtn = document.getElementById('change-doc-btn');
  
  const askForm = document.getElementById('ask-form');
  const questionInput = document.getElementById('question-input');
  const sendBtn = document.getElementById('send-btn');
  const messagesContainer = document.getElementById('messages-container');
  const clearChatBtn = document.getElementById('clear-chat-btn');
  const suggestionChips = document.getElementById('suggestion-chips');
  const toastContainer = document.getElementById('toast-container');
  const systemStatusText = document.getElementById('system-status-text');
  const systemStatusDot = document.getElementById('system-status-dot');

  // Initialize Lucide Icons
  lucide.createIcons();

  // Configure marked for clean markdown rendering
  if (window.marked) {
    marked.setOptions({
      breaks: true,
      gfm: true
    });
  }

  // 1. Initial State Check from Backend
  checkInitialStatus();

  async function checkInitialStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        if (data.has_document && data.document) {
          showActiveDocument(data.document);
        } else {
          setEngineStatus('ready', 'Engine Ready');
        }
      }
    } catch (e) {
      console.log('Status check error:', e);
      setEngineStatus('ready', 'Engine Online');
    }
  }

  function setEngineStatus(state, label) {
    systemStatusText.innerText = label;
    if (state === 'active') {
      systemStatusDot.classList.add('active');
    } else {
      systemStatusDot.classList.remove('active');
    }
  }

  // 2. Drag and Drop Ingestion Handlers
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  });

  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // 3. Document File Upload & Ingestion
  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    startUploadProgress(`Ingesting: ${file.name}`);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (res.ok && data.success) {
        completeUploadProgress(() => {
          showActiveDocument(data.document);
          showToast(`Successfully indexed '${file.name}' into vector store.`, "success");
        });
      } else {
        resetUploadProgress();
        showToast(data.detail || "Failed to parse and index document.", "error");
      }
    } catch (err) {
      resetUploadProgress();
      showToast("Server connection error during upload.", "error");
    }
  }

  let progressTicker = null;

  function startUploadProgress(titleText) {
    uploadProgressCard.classList.remove('hidden');
    progressStepText.innerText = titleText;
    progressBarFill.style.width = '25%';
    progressPercent.innerText = 'Extracting...';
    progressSubdetail.innerText = 'Extracting text tokens and constructing chunks...';

    let current = 25;
    progressTicker = setInterval(() => {
      if (current < 90) {
        current += Math.floor(Math.random() * 12) + 4;
        if (current > 90) current = 90;
        progressBarFill.style.width = `${current}%`;
        
        if (current > 50 && current < 75) {
          progressPercent.innerText = 'Embedding...';
          progressSubdetail.innerText = 'Computing high-dimensional OpenAI vector embeddings...';
        } else if (current >= 75) {
          progressPercent.innerText = 'Indexing...';
          progressSubdetail.innerText = 'Writing vector index into Chroma in-memory store...';
        }
      }
    }, 280);
  }

  function completeUploadProgress(callback) {
    clearInterval(progressTicker);
    progressBarFill.style.width = '100%';
    progressPercent.innerText = 'Complete';
    progressStepText.innerText = "Document Index Ready";
    progressSubdetail.innerText = "Vector index successfully constructed.";

    setTimeout(() => {
      resetUploadProgress();
      if (callback) callback();
    }, 450);
  }

  function resetUploadProgress() {
    clearInterval(progressTicker);
    uploadProgressCard.classList.add('hidden');
    progressBarFill.style.width = '0%';
    progressPercent.innerText = 'Processing';
    fileInput.value = '';
  }

  // 4. Reveal Active Retrieval Workspace
  function showActiveDocument(doc) {
    activeDocFilename.innerText = doc.filename || "Uploaded Document";
    metricChunks.innerText = doc.chunk_count || 0;
    metricPages.innerText = doc.page_count || 1;
    metricChars.innerText = (doc.total_chars || 0).toLocaleString();

    setEngineStatus('active', `Index Active (${doc.chunk_count} Chunks)`);

    uploadSection.classList.add('hidden');
    chatSection.classList.remove('hidden');

    lucide.createIcons();

    setTimeout(() => {
      questionInput.focus();
    }, 200);
  }

  // 5. Change / Reset Document
  changeDocBtn.addEventListener('click', async () => {
    try {
      await fetch('/api/reset', { method: 'POST' });
    } catch (e) {}

    chatSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    setEngineStatus('ready', 'Engine Ready');
    showToast("Index purged. Please upload a document to proceed.", "info");
    lucide.createIcons();
  });

  // 6. Textarea Resizing and Keybinds
  questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + 'px';
  });

  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    }
  });

  // 7. Suggestion Chips
  suggestionChips.addEventListener('click', (e) => {
    const chip = e.target.closest('.query-chip');
    if (chip) {
      const prompt = chip.getAttribute('data-prompt');
      questionInput.value = prompt;
      questionInput.style.height = 'auto';
      questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + 'px';
      askForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    }
  });

  // 8. Submit Query Handler
  askForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    // Render User Query
    appendUserMessage(question);
    questionInput.value = '';
    questionInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Render Skeleton Loader
    const loadingId = appendLoadingMessage();

    try {
      const startTime = performance.now();
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });

      const data = await res.json();
      const durationSec = ((performance.now() - startTime) / 1000).toFixed(2);
      removeLoadingMessage(loadingId);

      if (res.ok) {
        if (!data.time_taken_sec) {
          data.time_taken_sec = durationSec;
        }
        appendAssistantMessage(data);
      } else {
        appendAssistantMessage({
          answer: data.detail || "An error occurred during vector retrieval.",
          source: "error",
          chunks: []
        });
      }
    } catch (err) {
      removeLoadingMessage(loadingId);
      appendAssistantMessage({
        answer: "Could not establish connection with retrieval backend. Ensure server is active.",
        source: "error",
        chunks: []
      });
    } finally {
      sendBtn.disabled = false;
      questionInput.focus();
    }
  });

  // 9. Message Construction & DOM Injection
  function appendUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'console-message user-msg';
    msgDiv.innerHTML = `
      <div class="msg-avatar">
        <i data-lucide="user" class="avatar-icon"></i>
      </div>
      <div class="msg-content-wrapper">
        <div class="msg-header">
          <span class="sender-title">User Query</span>
        </div>
        <div class="msg-body">
          <p>${escapeHtml(text)}</p>
        </div>
      </div>
    `;
    messagesContainer.appendChild(msgDiv);
    lucide.createIcons();
    scrollToBottom();
  }

  function appendLoadingMessage() {
    const id = 'loading-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.id = id;
    msgDiv.className = 'console-message assistant-msg';
    msgDiv.innerHTML = `
      <div class="msg-avatar">
        <i data-lucide="cpu" class="avatar-icon"></i>
      </div>
      <div class="msg-content-wrapper">
        <div class="msg-header">
          <span class="sender-title">Ubaid's Retrieval Assistant</span>
          <span class="badge-blue"><i data-lucide="loader" class="spin-icon"></i> Searching Chroma Vector Space...</span>
        </div>
        <div class="loading-body">
          <div class="skeleton-row w-90"></div>
          <div class="skeleton-row w-75"></div>
          <div class="skeleton-row w-50"></div>
        </div>
      </div>
    `;
    messagesContainer.appendChild(msgDiv);
    lucide.createIcons();
    scrollToBottom();
    return id;
  }

  function removeLoadingMessage(id) {
    const elem = document.getElementById(id);
    if (elem) elem.remove();
  }

  function appendAssistantMessage(data) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'console-message assistant-msg';

    const isDoc = data.source === 'document';
    const isDocNotFound = data.source === 'document_not_found';
    const timeTaken = data.time_taken_sec ? `${data.time_taken_sec}s` : '';

    let badgeHtml = '';
    if (isDoc) {
      badgeHtml = `<span class="badge-blue"><i data-lucide="file-check"></i> Document Grounded</span>`;
    } else if (isDocNotFound) {
      badgeHtml = `<span class="badge-red"><i data-lucide="alert-circle"></i> Unmatched in Context</span>`;
    } else {
      badgeHtml = `<span class="badge-red"><i data-lucide="alert-triangle"></i> System Notice</span>`;
    }

    const formattedAnswer = window.marked ? marked.parse(data.answer) : `<p>${escapeHtml(data.answer)}</p>`;

    msgDiv.innerHTML = `
      <div class="msg-avatar">
        <i data-lucide="cpu" class="avatar-icon"></i>
      </div>
      <div class="msg-content-wrapper">
        <div class="msg-header">
          <span class="sender-title">Ubaid's Retrieval Assistant</span>
          ${badgeHtml}
          ${timeTaken ? `<span class="meta-latency"><i data-lucide="clock"></i> ${timeTaken}</span>` : ''}
        </div>
        <div class="msg-body">
          ${formattedAnswer}
        </div>
        <div class="msg-actions">
          <button class="action-btn-pill copy-btn" title="Copy answer text">
            <i data-lucide="copy"></i> Copy Text
          </button>
        </div>
      </div>
    `;

    messagesContainer.appendChild(msgDiv);

    // Copy Handler
    const copyBtn = msgDiv.querySelector('.copy-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(data.answer);
        showToast("Answer copied to clipboard.", "info");
      });
    }

    lucide.createIcons();
    scrollToBottom();
  }

  // 10. Clear Chat History
  clearChatBtn.addEventListener('click', () => {
    messagesContainer.innerHTML = `
      <div class="console-message assistant-msg initial-msg">
        <div class="msg-avatar">
          <i data-lucide="cpu" class="avatar-icon"></i>
        </div>
        <div class="msg-content-wrapper">
          <div class="msg-header">
            <span class="sender-title">Ubaid's Retrieval Assistant</span>
            <span class="badge-blue"><i data-lucide="check"></i> Ready for Querying</span>
          </div>
          <div class="msg-body">
            <p>Chat history cleared. Submit any question below to retrieve grounded answers from the active document.</p>
          </div>
        </div>
      </div>
    `;
    lucide.createIcons();
    showToast("Console history cleared.", "info");
  });

  // 11. Utilities
  function scrollToBottom() {
    setTimeout(() => {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }, 40);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast-item toast-${type}`;
    let iconName = 'info';
    if (type === 'success') iconName = 'check-circle';
    if (type === 'error') iconName = 'alert-triangle';

    toast.innerHTML = `
      <i data-lucide="${iconName}"></i>
      <span>${escapeHtml(message)}</span>
    `;

    toastContainer.appendChild(toast);
    lucide.createIcons();

    setTimeout(() => {
      toast.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }
});

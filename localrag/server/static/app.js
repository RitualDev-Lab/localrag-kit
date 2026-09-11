// LocalRAG-Kit Frontend Application

document.addEventListener("DOMContentLoaded", () => {
  // State
  let allFiles = [];
  let currentCitations = [];

  // DOM Elements - Navigation & Tabs
  const tabChatBtn = document.getElementById("tab-chat-btn");
  const tabSearchBtn = document.getElementById("tab-search-btn");
  const tabFilesBtn = document.getElementById("tab-files-btn");
  const viewChat = document.getElementById("view-chat");
  const viewSearch = document.getElementById("view-search");
  const sidebar = document.getElementById("sidebar");

  // DOM Elements - Workspace & Status
  const workspaceLabel = document.getElementById("workspace-label");
  const statsBadge = document.getElementById("stats-badge");
  const modelBadge = document.getElementById("model-badge");
  const embedBadge = document.getElementById("embed-badge");
  const reindexBtn = document.getElementById("reindex-btn");
  const reindexIcon = document.getElementById("reindex-icon");

  // DOM Elements - File Catalog
  const fileFilterInput = document.getElementById("file-filter-input");
  const fileCountLabel = document.getElementById("file-count-label");
  const fileList = document.getElementById("file-list");

  // DOM Elements - Chat
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const selectMode = document.getElementById("select-mode");
  const messagesContainer = document.getElementById("messages-container");

  // DOM Elements - Search
  const searchForm = document.getElementById("search-form");
  const searchQueryInput = document.getElementById("search-query-input");
  const searchModeSelect = document.getElementById("search-mode-select");
  const searchResultsContainer = document.getElementById("search-results-container");

  // DOM Elements - Source Drawer
  const sourceDrawer = document.getElementById("source-drawer");
  const closeDrawerBtn = document.getElementById("close-drawer-btn");
  const drawerSourceBadge = document.getElementById("drawer-source-badge");
  const drawerFilename = document.getElementById("drawer-filename");
  const drawerLines = document.getElementById("drawer-lines");
  const drawerScore = document.getElementById("drawer-score");
  const drawerCode = document.getElementById("drawer-code");

  // DOM Elements - Toast
  const toast = document.getElementById("toast");
  const toastText = document.getElementById("toast-text");

  // Initialize Lucide icons
  if (window.lucide) {
    window.lucide.createIcons();
  }

  // Toast Notification
  function showToast(msg, isError = false) {
    toastText.textContent = msg;
    if (isError) {
      toast.classList.remove("border-slate-700");
      toast.classList.add("border-rose-500", "text-rose-200");
    } else {
      toast.classList.remove("border-rose-500", "text-rose-200");
      toast.classList.add("border-slate-700", "text-white");
    }
    toast.classList.remove("opacity-0", "pointer-events-none");
    toast.classList.add("opacity-100");
    setTimeout(() => {
      toast.classList.remove("opacity-100");
      toast.classList.add("opacity-0", "pointer-events-none");
    }, 3500);
  }

  // Tab Switching
  function switchTab(tab) {
    if (tab === "chat") {
      viewChat.classList.remove("hidden");
      viewSearch.classList.add("hidden");
      tabChatBtn.className = "px-3 py-1.5 rounded-md font-medium text-white bg-slate-700 shadow-sm flex items-center gap-1.5 transition-colors";
      tabSearchBtn.className = "px-3 py-1.5 rounded-md font-medium text-slate-400 hover:text-white flex items-center gap-1.5 transition-colors";
    } else if (tab === "search") {
      viewChat.classList.add("hidden");
      viewSearch.classList.remove("hidden");
      tabSearchBtn.className = "px-3 py-1.5 rounded-md font-medium text-white bg-slate-700 shadow-sm flex items-center gap-1.5 transition-colors";
      tabChatBtn.className = "px-3 py-1.5 rounded-md font-medium text-slate-400 hover:text-white flex items-center gap-1.5 transition-colors";
      searchQueryInput.focus();
    } else if (tab === "files") {
      sidebar.classList.toggle("hidden");
    }
  }

  tabChatBtn.addEventListener("click", () => switchTab("chat"));
  tabSearchBtn.addEventListener("click", () => switchTab("search"));
  if (tabFilesBtn) {
    tabFilesBtn.addEventListener("click", () => switchTab("files"));
  }

  // Fetch Status
  async function loadStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error("Failed to fetch status");
      const data = await res.json();

      workspaceLabel.textContent = data.workspace_path || "Workspace";
      workspaceLabel.title = data.workspace_path;
      const totalFiles = data.stats ? data.stats.total_files : 0;
      const totalChunks = data.stats ? data.stats.total_chunks : 0;
      statsBadge.textContent = `${totalFiles} files · ${totalChunks} chunks`;
      modelBadge.textContent = `LLM: ${data.llm_provider || "Auto"}`;
      embedBadge.textContent = `Embedder: ${data.embedding_provider || "Fast"}`;
    } catch (err) {
      console.error("Status error:", err);
      statsBadge.textContent = "Offline";
    }
  }

  // Fetch Files
  async function loadFiles() {
    try {
      const res = await fetch("/api/files");
      if (!res.ok) throw new Error("Failed to fetch files");
      const data = await res.json();
      allFiles = data.files || [];
      renderFiles(allFiles);
    } catch (err) {
      console.error("Files error:", err);
    }
  }

  function renderFiles(files) {
    fileCountLabel.textContent = files.length;
    fileList.innerHTML = "";

    if (files.length === 0) {
      fileList.innerHTML = `<div class="p-3 text-xs text-slate-500 italic">No files indexed yet.</div>`;
      return;
    }

    files.forEach(f => {
      const item = document.createElement("div");
      item.className = "p-2 hover:bg-slate-800/60 rounded-lg cursor-pointer transition-colors group flex items-start justify-between";
      
      const pathStr = f.relative_path || f.path || "";
      const ext = pathStr.slice(pathStr.lastIndexOf(".")).toLowerCase();
      const isCode = [".py", ".ts", ".js", ".go", ".rs", ".java", ".c", ".cpp"].includes(ext);
      const iconName = isCode ? "code" : (ext === ".pdf" ? "file-text" : "file");

      item.innerHTML = `
        <div class="flex items-start gap-2 truncate min-w-0 pr-2">
          <i data-lucide="${iconName}" class="w-3.5 h-3.5 text-slate-500 group-hover:text-brand-400 mt-0.5 shrink-0"></i>
          <span class="text-xs text-slate-300 group-hover:text-white truncate" title="${pathStr}">${pathStr}</span>
        </div>
        <span class="text-[10px] text-slate-500 shrink-0 tabular-nums">${f.line_count || 0} lines</span>
      `;

      item.addEventListener("click", () => {
        searchQueryInput.value = pathStr;
        switchTab("search");
        performSearch(pathStr);
      });

      fileList.appendChild(item);
    });

    if (window.lucide) window.lucide.createIcons();
  }

  fileFilterInput.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase().trim();
    const filtered = allFiles.filter(f => {
      const p = (f.relative_path || f.path || "").toLowerCase();
      return p.includes(query);
    });
    renderFiles(filtered);
  });

  // Reindex Action
  reindexBtn.addEventListener("click", async () => {
    reindexBtn.disabled = true;
    reindexIcon.classList.add("spin-icon");
    showToast("Re-indexing workspace...");

    try {
      const res = await fetch("/api/reindex", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full: false, embed: true })
      });
      const data = await res.json();
      if (res.ok) {
        const secs = (data.time_taken_ms / 1000).toFixed(2);
        showToast(`Indexed ${data.files_scanned} files (${data.chunks_indexed} chunks) in ${secs}s`);
        await loadStatus();
        await loadFiles();
      } else {
        showToast(data.detail || "Re-indexing failed", true);
      }
    } catch (err) {
      showToast("Error connecting to server", true);
    } finally {
      reindexBtn.disabled = false;
      reindexIcon.classList.remove("spin-icon");
    }
  });

  // Chat Form & Streaming
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 144) + "px";
  });

  function appendUserMessage(text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "flex gap-4 max-w-3xl mx-auto justify-end";
    msgDiv.innerHTML = `
      <div class="flex-1 space-y-1 text-right">
        <p class="font-semibold text-xs text-slate-400">You</p>
        <div class="inline-block bg-brand-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm max-w-xl text-left shadow-md">
          <p class="whitespace-pre-wrap">${escapeHtml(text)}</p>
        </div>
      </div>
      <div class="w-8 h-8 rounded-lg bg-slate-800 text-slate-300 border border-slate-700 flex items-center justify-center shrink-0">
        <i data-lucide="user" class="w-4 h-4"></i>
      </div>
    `;
    messagesContainer.appendChild(msgDiv);
    if (window.lucide) window.lucide.createIcons();
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function appendAssistantMessage() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "flex gap-4 max-w-3xl mx-auto assistant-msg";
    msgDiv.innerHTML = `
      <div class="w-8 h-8 rounded-lg bg-brand-500/20 text-brand-400 border border-brand-500/30 flex items-center justify-center shrink-0 mt-1">
        <i data-lucide="bot" class="w-4 h-4"></i>
      </div>
      <div class="flex-1 space-y-2 min-w-0">
        <div class="flex items-center gap-2">
          <p class="font-semibold text-sm text-white">LocalRAG Assistant</p>
          <span class="text-[10px] text-slate-500 font-mono time-stamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="prose prose-invert prose-sm text-slate-200 leading-relaxed content-body">
          <span class="streaming-cursor"></span>
        </div>
        <div class="citations-container pt-2 flex flex-wrap gap-1.5"></div>
      </div>
    `;
    messagesContainer.appendChild(msgDiv);
    if (window.lucide) window.lucide.createIcons();
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return msgDiv;
  }

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const queryText = chatInput.value.trim();
    if (!queryText) return;

    appendUserMessage(queryText);
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;

    const assistantMsgElem = appendAssistantMessage();
    const contentBody = assistantMsgElem.querySelector(".content-body");
    const citationsContainer = assistantMsgElem.querySelector(".citations-container");
    let accumulatedText = "";
    let localCitations = [];

    const mode = selectMode.value;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText, mode: mode, top_k: 6 }),
      });

      if (!response.ok) {
        throw new Error(`HTTP Error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // Keep incomplete chunk in buffer

        for (const rawBlock of lines) {
          const trimmed = rawBlock.trim();
          if (!trimmed.startsWith("data: ")) continue;

          try {
            const dataObj = JSON.parse(trimmed.slice(6));
            if (dataObj.type === "citations") {
              localCitations = dataObj.citations || [];
              currentCitations = localCitations;
            } else if (dataObj.type === "token") {
              accumulatedText += dataObj.token || "";
              renderMarkdown(contentBody, accumulatedText, true);
              messagesContainer.scrollTop = messagesContainer.scrollHeight;
            } else if (dataObj.type === "error") {
              accumulatedText += `\n\n**Error:** ${dataObj.message}`;
              renderMarkdown(contentBody, accumulatedText, false);
            }
          } catch (pe) {
            console.error("JSON parse error in SSE chunk:", pe);
          }
        }
      }

      // Finalize Markdown Rendering
      renderMarkdown(contentBody, accumulatedText, false);

      // Render Citations Badges
      if (localCitations && localCitations.length > 0) {
        citationsContainer.innerHTML = `<span class="text-[11px] text-slate-400 font-semibold mr-1 self-center">Sources:</span>`;
        localCitations.forEach((cit, idx) => {
          const badge = document.createElement("button");
          badge.className = "citation-badge";
          badge.innerHTML = `<i data-lucide="file-text" class="w-3 h-3"></i> [${cit.source_index || (idx + 1)}] ${cit.relative_path}:${cit.start_line}`;
          badge.addEventListener("click", () => openDrawer(cit, cit.source_index || (idx + 1)));
          citationsContainer.appendChild(badge);
        });
        if (window.lucide) window.lucide.createIcons();
      }

    } catch (err) {
      console.error("Chat error:", err);
      contentBody.innerHTML = `<p class="text-rose-400">Failed to connect to local assistant: ${err.message}</p>`;
    } finally {
      sendBtn.disabled = false;
      chatInput.focus();
    }
  });

  function renderMarkdown(element, markdownText, isStreaming) {
    if (!window.marked) {
      element.textContent = markdownText;
      return;
    }
    const html = window.marked.parse(markdownText);
    element.innerHTML = html + (isStreaming ? '<span class="streaming-cursor"></span>' : '');
    
    // Highlight code blocks
    if (window.hljs) {
      element.querySelectorAll("pre code").forEach((block) => {
        window.hljs.highlightElement(block);
      });
    }

    // Replace inline citation markers [Source #N] with clickable badges
    if (!isStreaming) {
      element.innerHTML = element.innerHTML.replace(/\[Source #(\d+)\]/g, (match, p1) => {
        const idx = parseInt(p1, 10);
        return `<button class="citation-badge" data-cit-idx="${idx}">[Source #${idx}]</button>`;
      });

      element.querySelectorAll("button[data-cit-idx]").forEach(btn => {
        btn.addEventListener("click", () => {
          const idx = parseInt(btn.getAttribute("data-cit-idx"), 10);
          const found = currentCitations.find(c => c.source_index === idx) || currentCitations[idx - 1];
          if (found) {
            openDrawer(found, idx);
          }
        });
      });
    }
  }

  // Search Implementation
  async function performSearch(query) {
    if (!query) return;
    const mode = searchModeSelect.value;
    searchResultsContainer.innerHTML = `
      <div class="text-center py-12 text-slate-400 text-sm">
        <i data-lucide="loader" class="w-6 h-6 mx-auto mb-2 spin-icon text-brand-400"></i>
        Searching indexed chunks (${mode})...
      </div>
    `;
    if (window.lucide) window.lucide.createIcons();

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, mode: mode, limit: 12 }),
      });

      if (!res.ok) throw new Error("Search request failed");
      const data = await res.json();
      renderSearchResults(data.results || [], data.query, data.mode);
    } catch (err) {
      searchResultsContainer.innerHTML = `
        <div class="p-4 bg-rose-950/40 border border-rose-800/60 rounded-xl text-rose-300 text-sm">
          Failed to perform search: ${err.message}
        </div>
      `;
    }
  }

  searchForm.addEventListener("submit", (e) => {
    e.preventDefault();
    performSearch(searchQueryInput.value.trim());
  });

  function renderSearchResults(results, query, mode) {
    if (results.length === 0) {
      searchResultsContainer.innerHTML = `
        <div class="text-center py-16 text-slate-500 text-sm">
          No matching chunks found for "${escapeHtml(query)}" in ${mode} mode.
        </div>
      `;
      return;
    }

    searchResultsContainer.innerHTML = `
      <div class="flex justify-between items-center text-xs text-slate-400 px-1 pb-2">
        <span>Found <strong>${results.length}</strong> chunks matching "${escapeHtml(query)}"</span>
        <span class="uppercase tracking-wider font-semibold text-[10px] bg-slate-800 px-2 py-0.5 rounded border border-slate-700">${mode}</span>
      </div>
    `;

    results.forEach((item, idx) => {
      const card = document.createElement("div");
      card.className = "p-4 bg-slate-900/60 border border-slate-800 hover:border-brand-500/50 rounded-xl transition-all space-y-3 cursor-pointer group shadow-sm";
      
      const chunk = item.chunk || {};
      const meta = chunk.metadata || {};
      const relPath = meta.relative_path || "file";
      const startLine = meta.start_line || 1;
      const endLine = meta.end_line || 1;
      const text = chunk.text || "";
      const scoreStr = item.score !== undefined ? item.score.toFixed(4) : "N/A";
      
      card.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2 truncate">
            <span class="px-2 py-0.5 bg-brand-500/10 text-brand-400 border border-brand-500/20 rounded text-[11px] font-mono font-bold">#${idx + 1}</span>
            <span class="font-medium text-sm text-slate-200 group-hover:text-white truncate">${relPath}</span>
          </div>
          <div class="flex items-center gap-3 text-xs text-slate-400 shrink-0">
            <span>Lines ${startLine} - ${endLine}</span>
            <span class="font-mono text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/40">Score: ${scoreStr}</span>
          </div>
        </div>
        <pre class="bg-slate-950 p-3 rounded-lg text-xs font-mono text-slate-300 overflow-x-auto max-h-32 custom-scrollbar border border-slate-900"><code>${escapeHtml(text)}</code></pre>
      `;

      card.addEventListener("click", () => {
        openDrawer({
          relative_path: relPath,
          start_line: startLine,
          end_line: endLine,
          text_snippet: text,
          score: item.score
        }, idx + 1);
      });

      searchResultsContainer.appendChild(card);
    });

    if (window.hljs) {
      searchResultsContainer.querySelectorAll("pre code").forEach(block => {
        window.hljs.highlightElement(block);
      });
    }
  }

  // Source Drawer Control
  function openDrawer(source, indexNum) {
    drawerSourceBadge.textContent = `Source #${indexNum || 1}`;
    drawerFilename.textContent = source.relative_path || source.file_path || "File";
    drawerLines.textContent = `Lines ${source.start_line} - ${source.end_line}`;
    drawerScore.textContent = source.score !== undefined ? `Score: ${source.score.toFixed(4)}` : "";
    drawerCode.textContent = source.text_snippet || source.snippet || source.text || "";

    sourceDrawer.classList.remove("translate-x-full");

    if (window.hljs) {
      window.hljs.highlightElement(drawerCode);
    }
  }

  function closeDrawer() {
    sourceDrawer.classList.add("translate-x-full");
  }

  closeDrawerBtn.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
  });

  // Helper: Escape HTML
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Initial Load
  loadStatus();
  loadFiles();
});

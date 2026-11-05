import React, { useState, useEffect, useRef } from 'react';
import { marked } from 'marked';
import {
  ShieldCheck,
  AlertCircle,
  Terminal,
  ChevronDown,
  ChevronUp,
  Zap,
  Clock,
  ArrowRight,
  Database,
  CheckCircle2,
  GitBranch,
  Plus,
  RefreshCw,
  X,
  Github,
  Layers,
  Code2,
} from 'lucide-react';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [retriever, setRetriever] = useState('hybrid');
  const [variant, setVariant] = useState('v2');
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);

  // Repository state
  const [repos, setRepos] = useState([]);
  const [selectedRepo, setSelectedRepo] = useState('StreamNova');
  const [showIndexModal, setShowIndexModal] = useState(false);
  const [newRepoUrl, setNewRepoUrl] = useState('');
  const [newRepoVariant, setNewRepoVariant] = useState('v2');
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState(null);
  const [indexError, setIndexError] = useState(null);

  const [expandedCitations, setExpandedCitations] = useState({});
  const [showChunks, setShowChunks] = useState({});
  const chatEndRef = useRef(null);

  // Fetch system health on mount
  useEffect(() => {
    fetch('/health')
      .then((res) => res.json())
      .then((data) => setHealth(data))
      .catch((err) => console.error('Health check error:', err));
  }, []);

  // Fetch registered repositories on mount
  const fetchRepos = async () => {
    try {
      const res = await fetch('/repos');
      if (res.ok) {
        const data = await res.json();
        setRepos(data);
        if (data.length > 0 && !selectedRepo) {
          setSelectedRepo(data[0].repo_id);
        }
      }
    } catch (err) {
      console.error('Error fetching repositories:', err);
    }
  };

  useEffect(() => {
    fetchRepos();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const currentRepoObj = repos.find((r) => r.repo_id === selectedRepo) || {
    repo_id: selectedRepo || 'StreamNova',
    repo_name: selectedRepo === 'StreamNova' ? 'KaranPushes/StreamNova' : (selectedRepo || 'Repository'),
    total_chunks: repos.find((r) => r.repo_id === selectedRepo)?.total_chunks || 1336,
  };

  const handleAsk = async (e) => {
    e?.preventDefault();
    if (!question.trim() || loading) return;

    const userQ = question.trim();
    setQuestion('');

    setMessages((prev) => [...prev, { role: 'user', content: userQ }]);
    setLoading(true);

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userQ,
          repo_id: selectedRepo,
          retriever,
          variant,
          k: parseInt(k, 10),
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(err.detail || 'Server error');
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          answer: data.answer,
          citations: data.citations || [],
          retrieved_chunks: data.retrieved_chunks || [],
          latency_ms: data.latency_ms,
          verification_rate: data.verification_rate,
          retriever: retriever,
          repo_id: selectedRepo,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          error: err.message,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleIndexRepo = async (e) => {
    e?.preventDefault();
    if (!newRepoUrl.trim() || isIndexing) return;

    setIsIndexing(true);
    setIndexError(null);
    setIndexStatus({ message: 'Cloning repository and computing code embeddings... This may take ~30-60s.' });

    try {
      const res = await fetch('/index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_url: newRepoUrl.trim(),
          variant: newRepoVariant,
          reindex: false,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Failed to index repository');
      }

      setIndexStatus({
        success: true,
        summary: data.summary,
        message: data.message,
      });

      await fetchRepos();

      if (data.summary?.repo_id) {
        setSelectedRepo(data.summary.repo_id);
      }

      setNewRepoUrl('');
    } catch (err) {
      setIndexError(err.message);
      setIndexStatus(null);
    } finally {
      setIsIndexing(false);
    }
  };

  const toggleCitation = (msgIndex, citIndex) => {
    const key = `${msgIndex}-${citIndex}`;
    setExpandedCitations((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleChunks = (msgIndex) => {
    setShowChunks((prev) => ({ ...prev, [msgIndex]: !prev[msgIndex] }));
  };

  const sampleQueries = selectedRepo.includes('StreamNova')
    ? [
        'How does JWT authentication work in this platform?',
        'Where is stream scheduling handled?',
        'Where is WebSocket chat message broadcasting implemented?',
      ]
    : [
        `What is the overall architecture and main entry point of this codebase?`,
        `How is request routing or handling implemented?`,
        `Where are authentication, configuration, or data models handled?`,
      ];

  return (
    <div className="flex flex-col h-screen max-w-7xl mx-auto w-full p-4 md:p-6 text-zinc-100">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-zinc-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center font-bold text-zinc-100 shadow-sm">
            <Terminal className="w-5 h-5 text-zinc-200" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-zinc-100 flex items-center gap-2">
              RepoChat
              <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-zinc-900 text-zinc-300 border border-zinc-800 flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                Verified Citations
              </span>
            </h1>
            <p className="text-xs text-zinc-400">
              Grounded Codebase RAG with Line-Level Verification
            </p>
          </div>
        </div>

        {/* Repository Switcher & Index New Button */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Active Repo Dropdown */}
          <div className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 px-3 py-1.5 rounded-lg text-xs transition-colors">
            <GitBranch className="w-4 h-4 text-zinc-400 shrink-0" />
            <span className="text-zinc-500">Repo:</span>
            <select
              value={selectedRepo}
              onChange={(e) => setSelectedRepo(e.target.value)}
              className="bg-transparent text-zinc-200 font-semibold focus:outline-none cursor-pointer pr-1"
            >
              {repos.length > 0 ? (
                repos.map((r) => (
                  <option key={r.repo_id} value={r.repo_id} className="bg-zinc-900 text-zinc-200">
                    {r.repo_name} ({r.total_chunks} chunks)
                  </option>
                ))
              ) : (
                <option value="StreamNova" className="bg-zinc-900 text-zinc-200">
                  StreamNova (1,336 chunks)
                </option>
              )}
            </select>
          </div>

          {/* Index New GitHub Repo Button */}
          <button
            onClick={() => {
              setShowIndexModal(true);
              setIndexError(null);
              setIndexStatus(null);
            }}
            className="flex items-center gap-1.5 bg-zinc-850 hover:bg-zinc-800 text-zinc-200 border border-zinc-750 text-xs font-medium px-3 py-1.5 rounded-lg shadow-sm transition-all"
          >
            <Plus className="w-3.5 h-3.5 text-zinc-300" />
            <span>Index GitHub Repo</span>
          </button>

          {/* System Health Status */}
          <div className="hidden sm:flex items-center gap-2 bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded-lg text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                health?.status === 'ok' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span className="text-zinc-400">
              {health ? `${health.embedding_provider} / ${health.generation_model}` : 'Connecting...'}
            </span>
          </div>
        </div>
      </header>

      {/* Control Ribbon */}
      <div className="flex flex-wrap items-center justify-between gap-4 py-3 border-b border-zinc-800/80 text-xs">
        <div className="flex flex-wrap items-center gap-4">
          {/* Strategy */}
          <div className="flex items-center gap-2">
            <label className="text-zinc-400 font-medium">Retriever:</label>
            <select
              value={retriever}
              onChange={(e) => setRetriever(e.target.value)}
              className="bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-md px-2.5 py-1 text-xs focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 outline-none"
            >
              <option value="hybrid">Hybrid (Vector + BM25 RRF)</option>
              <option value="vector">Vector Only (Dense Semantic)</option>
              <option value="bm25">BM25 Only (Exact Keyword)</option>
            </select>
          </div>

          {/* Variant */}
          <div className="flex items-center gap-2">
            <label className="text-zinc-400 font-medium">Variant:</label>
            <select
              value={variant}
              onChange={(e) => setVariant(e.target.value)}
              className="bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-md px-2.5 py-1 text-xs focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 outline-none"
            >
              <option value="v2">v2 (Context Header & Scope)</option>
              <option value="v1">v1 (Plain Chunk Text)</option>
            </select>
          </div>

          {/* K */}
          <div className="flex items-center gap-2">
            <label className="text-zinc-400 font-medium">Top-K:</label>
            <select
              value={k}
              onChange={(e) => setK(e.target.value)}
              className="bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-md px-2.5 py-1 text-xs focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 outline-none"
            >
              <option value="3">3 Chunks</option>
              <option value="5">5 Chunks</option>
              <option value="8">8 Chunks</option>
            </select>
          </div>
        </div>

        <div className="text-zinc-400 flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-zinc-400" />
            <span className="text-zinc-500">Target:</span>
            <span className="text-zinc-200 font-mono font-medium">{currentRepoObj.repo_name}</span>
          </div>
          <span className="text-zinc-800">|</span>
          <div className="flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-zinc-400" />
            <span className="text-zinc-500">Strategy:</span>
            <span className="text-zinc-200 font-semibold">{retriever.toUpperCase()}</span>
          </div>
        </div>
      </div>

      {/* Index GitHub Repo Modal */}
      {showIndexModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl relative space-y-4">
            <button
              onClick={() => {
                if (!isIndexing) setShowIndexModal(false);
              }}
              disabled={isIndexing}
              className="absolute top-4 right-4 text-zinc-400 hover:text-zinc-200 disabled:opacity-50"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-zinc-800 border border-zinc-700/80 flex items-center justify-center text-zinc-200">
                <Github className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-zinc-100">Index Any GitHub Repository</h3>
                <p className="text-xs text-zinc-400">
                  Clone, parse, chunk, and embed any public codebase into RepoChat.
                </p>
              </div>
            </div>

            <form onSubmit={handleIndexRepo} className="space-y-4 pt-2">
              <div>
                <label className="block text-xs font-semibold text-zinc-300 mb-1.5">
                  GitHub Repository Link or Clone URL
                </label>
                <input
                  type="text"
                  required
                  value={newRepoUrl}
                  onChange={(e) => setNewRepoUrl(e.target.value)}
                  placeholder="https://github.com/owner/repository (e.g. https://github.com/pallets/markupsafe)"
                  disabled={isIndexing}
                  className="w-full bg-zinc-950 border border-zinc-800 focus:border-zinc-600 rounded-xl px-3.5 py-2.5 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none"
                />
              </div>

              <div className="flex items-center justify-between gap-4">
                <div className="flex-1">
                  <label className="block text-xs font-semibold text-zinc-300 mb-1.5">
                    Chunking Variant
                  </label>
                  <select
                    value={newRepoVariant}
                    onChange={(e) => setNewRepoVariant(e.target.value)}
                    disabled={isIndexing}
                    className="w-full bg-zinc-950 border border-zinc-800 focus:border-zinc-600 rounded-xl px-3 py-2 text-xs text-zinc-200 focus:outline-none"
                  >
                    <option value="v2">v2 (Context Header & Scope - Recommended)</option>
                    <option value="v1">v1 (Plain Chunk Text)</option>
                  </select>
                </div>
              </div>

              {/* Status or Progress */}
              {indexStatus && (
                <div
                  className={`p-3.5 rounded-xl border text-xs flex flex-col gap-1.5 ${
                    indexStatus.success
                      ? 'bg-emerald-950/40 border-emerald-800/80 text-emerald-300'
                      : 'bg-zinc-800/60 border-zinc-700 text-zinc-300'
                  }`}
                >
                  <div className="flex items-center gap-2 font-semibold">
                    {indexStatus.success ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    ) : (
                      <RefreshCw className="w-4 h-4 text-zinc-300 animate-spin shrink-0" />
                    )}
                    <span>{indexStatus.message}</span>
                  </div>
                  {indexStatus.summary && (
                    <div className="text-[11px] text-zinc-300 grid grid-cols-2 gap-1 pt-1 border-t border-emerald-800/40 font-mono">
                      <span>Files scanned: {indexStatus.summary.total_files_scanned}</span>
                      <span>Files indexed: {indexStatus.summary.total_files_indexed}</span>
                      <span>Total chunks: {indexStatus.summary.total_chunks}</span>
                      <span>Time taken: {indexStatus.summary.elapsed_time_sec}s</span>
                    </div>
                  )}
                </div>
              )}

              {/* Error Alert */}
              {indexError && (
                <div className="p-3.5 rounded-xl border border-rose-800/60 bg-rose-950/40 text-rose-300 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                  <span>{indexError}</span>
                </div>
              )}

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowIndexModal(false)}
                  disabled={isIndexing}
                  className="px-4 py-2 text-xs text-zinc-400 hover:text-zinc-200 rounded-lg transition-colors"
                >
                  Close
                </button>
                <button
                  type="submit"
                  disabled={isIndexing || !newRepoUrl.trim()}
                  className="px-5 py-2 bg-zinc-100 hover:bg-white disabled:bg-zinc-800 disabled:text-zinc-600 text-zinc-950 text-xs font-semibold rounded-lg shadow-sm transition-all flex items-center gap-2"
                >
                  {isIndexing && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                  <span>{isIndexing ? 'Cloning & Indexing...' : 'Clone & Index Repo'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Chat Messages */}
      <main className="flex-1 overflow-y-auto py-6 space-y-6">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-xl mx-auto space-y-4">
            <div className="w-14 h-14 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-300 shadow-inner">
              <Code2 className="w-7 h-7 text-zinc-300" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-zinc-100">
                Ask anything about <span className="text-zinc-100 font-mono underline decoration-zinc-700 underline-offset-4">{currentRepoObj.repo_name}</span>
              </h2>
              <p className="text-sm text-zinc-400 mt-1">
                Every answer is grounded strictly in the indexed codebase with verified line-level citations.
              </p>
            </div>
            <div className="w-full pt-2">
              <p className="text-xs uppercase tracking-wider text-zinc-500 font-semibold mb-2">Suggested queries</p>
              <div className="flex flex-col gap-2">
                {sampleQueries.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => setQuestion(q)}
                    className="text-left text-xs bg-zinc-900 hover:bg-zinc-850 border border-zinc-800 hover:border-zinc-700 text-zinc-300 hover:text-zinc-100 px-3.5 py-2.5 rounded-xl transition-all flex items-center justify-between"
                  >
                    <span>{q}</span>
                    <ArrowRight className="w-3.5 h-3.5 text-zinc-500" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {messages.map((msg, mIdx) => (
          <div key={mIdx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
            {msg.role === 'user' ? (
              <div className="bg-zinc-800 border border-zinc-700/80 text-zinc-100 px-4 py-2.5 rounded-2xl rounded-tr-sm max-w-2xl text-sm shadow-sm">
                {msg.content}
              </div>
            ) : (
              <div className="w-full bg-zinc-900/90 border border-zinc-800 rounded-2xl p-5 space-y-4 shadow-xl">
                {msg.error ? (
                  <div className="text-rose-400 text-sm bg-rose-950/40 border border-rose-800/50 p-3 rounded-lg flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                    <span><strong>Error:</strong> {msg.error}</span>
                  </div>
                ) : (
                  <>
                    {/* Metrics Bar */}
                    <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-800 text-xs">
                      <div className="flex items-center gap-3">
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/50 text-emerald-400 border border-emerald-800/60 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                          {Math.round((msg.verification_rate || 0) * 100)}% Citations Verified
                        </span>
                        <span className="text-zinc-400 flex items-center gap-1">
                          <Clock className="w-3.5 h-3.5 text-zinc-500" />
                          Latency: <strong className="text-zinc-200">{Math.round(msg.latency_ms)}ms</strong>
                        </span>
                        <span className="text-zinc-400">
                          Retriever: <strong className="text-zinc-200">{msg.retriever?.toUpperCase()}</strong>
                        </span>
                      </div>

                      <button
                        onClick={() => toggleChunks(mIdx)}
                        className="text-xs text-zinc-400 hover:text-zinc-200 underline transition-colors"
                      >
                        {showChunks[mIdx] ? 'Hide Chunks Context ▲' : `View Retrieved Chunks (${msg.retrieved_chunks?.length || 0}) ▼`}
                      </button>
                    </div>

                    {/* Markdown Answer */}
                    <div
                      className="prose prose-invert max-w-none text-sm text-zinc-200 leading-relaxed space-y-2"
                      dangerouslySetInnerHTML={{ __html: marked.parse(msg.answer || '') }}
                    />

                    {/* Citations List */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="pt-3 border-t border-zinc-800 space-y-2">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                          Citations ({msg.citations.length}) — Click to view code snippet
                        </h4>
                        <div className="grid grid-cols-1 gap-2">
                          {msg.citations.map((cit, cIdx) => {
                            const isExp = expandedCitations[`${mIdx}-${cIdx}`];
                            return (
                              <div
                                key={cIdx}
                                className="border border-zinc-800 rounded-lg bg-zinc-950/70 hover:border-zinc-700/80 overflow-hidden transition-all"
                              >
                                <button
                                  onClick={() => toggleCitation(mIdx, cIdx)}
                                  className="w-full flex items-center justify-between p-2.5 text-xs text-left hover:bg-zinc-900/80 transition-colors"
                                >
                                  <div className="flex items-center gap-2 truncate">
                                    <span
                                      className={`w-2 h-2 rounded-full ${
                                        cit.verified ? 'bg-emerald-400' : 'bg-rose-500'
                                      }`}
                                    />
                                    <span className="font-mono text-zinc-200 truncate font-medium">
                                      {cit.file}:{cit.start}-{cit.end}
                                    </span>
                                  </div>
                                  <div className="flex items-center gap-3 shrink-0">
                                    <span
                                      className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                                        cit.verified
                                          ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/60'
                                          : 'bg-rose-950/60 text-rose-400 border border-rose-800/60'
                                      }`}
                                    >
                                      {cit.verified ? 'VERIFIED' : 'HALLUCINATED'}
                                    </span>
                                    {isExp ? (
                                      <ChevronUp className="w-4 h-4 text-zinc-400" />
                                    ) : (
                                      <ChevronDown className="w-4 h-4 text-zinc-400" />
                                    )}
                                  </div>
                                </button>

                                {isExp && (
                                  <div className="p-3 bg-zinc-950 border-t border-zinc-800 font-mono text-xs overflow-x-auto">
                                    <div className="text-[11px] text-zinc-400 pb-2 flex justify-between items-center">
                                      <span>{cit.reason}</span>
                                      <span className="text-zinc-500">
                                        Lines {cit.start} to {cit.end}
                                      </span>
                                    </div>
                                    <div className="bg-zinc-900/90 rounded border border-zinc-800 p-2.5 leading-5 text-zinc-300">
                                      {cit.snippet ? (
                                        cit.snippet.split('\n').map((line, lIdx) => (
                                          <div key={lIdx} className="flex">
                                            <span className="line-number w-10 shrink-0 text-zinc-500">
                                              {cit.start + lIdx}
                                            </span>
                                            <span className="pl-3 whitespace-pre">{line}</span>
                                          </div>
                                        ))
                                      ) : (
                                        <div className="text-zinc-500 italic">No snippet preview available</div>
                                      )}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Retrieved Chunks Inspector */}
                    {showChunks[mIdx] && msg.retrieved_chunks && (
                      <div className="pt-3 border-t border-zinc-800 space-y-3">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                          Retrieved Code Chunks Passed to Prompt
                        </h4>
                        <div className="space-y-2">
                          {msg.retrieved_chunks.map((chk, kIdx) => (
                            <div key={kIdx} className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-xs font-mono">
                              <div className="flex items-center justify-between pb-1.5 text-zinc-400 border-b border-zinc-900 mb-2">
                                <span className="text-zinc-200 font-bold">
                                  #{kIdx + 1} {chk.file_path}:{chk.start_line}-{chk.end_line}
                                </span>
                                <span>Score: {chk.score.toFixed(4)} | Scope: {chk.scope || 'N/A'}</span>
                              </div>
                              <pre className="overflow-x-auto text-zinc-300 max-h-40">{chk.raw_text}</pre>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-3 text-zinc-400 text-xs bg-zinc-900 p-3 rounded-xl border border-zinc-800 w-fit animate-pulse">
            <span className="w-2 h-2 rounded-full bg-zinc-400 animate-ping" />
            <span>Retrieving code chunks and verifying citations...</span>
          </div>
        )}
        <div ref={chatEndRef} />
      </main>

      {/* Footer / Input Box */}
      <footer className="pt-2 border-t border-zinc-800">
        <form onSubmit={handleAsk} className="relative">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleAsk();
              }
            }}
            rows="2"
            placeholder={`Ask about ${currentRepoObj.repo_name} (e.g. 'How does authentication work?'). Press Enter to submit.`}
            className="w-full bg-zinc-900 border border-zinc-800 focus:border-zinc-650 rounded-xl px-4 py-3 pr-24 text-sm text-zinc-100 placeholder-zinc-500 outline-none resize-none shadow-inner focus:ring-1 focus:ring-zinc-700"
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="absolute right-3 bottom-3 px-4 py-1.5 bg-zinc-100 hover:bg-white disabled:bg-zinc-800 disabled:text-zinc-600 text-zinc-950 text-xs font-semibold rounded-lg shadow-sm transition-all flex items-center gap-1.5"
          >
            {loading ? 'Searching...' : 'Ask →'}
          </button>
        </form>
        <div className="flex justify-between items-center text-[11px] text-zinc-500 px-1 pt-1.5">
          <span>Shift+Enter for newline</span>
          <span>All citations verified against real code boundaries</span>
        </div>
      </footer>
    </div>
  );
}

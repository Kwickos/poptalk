import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { Segment } from '../types';
import TranscriptLine from '../components/TranscriptLine';

function formatElapsedTime(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

const INLINE_BARS = 16;

interface Toast {
  id: number;
  message: string;
  type: 'success' | 'error' | 'loading';
}

interface SessionViewProps {
  onSessionStopped: (sessionId: string) => void;
  onLiveSessionChange: (id: string | null) => void;
  onLiveTextChange: (text: string) => void;
}

export default function SessionView({ onSessionStopped, onLiveSessionChange, onLiveTextChange }: SessionViewProps) {
  const [mode, setMode] = useState<'visio' | 'presentiel'>('visio');
  const [isRecording, setIsRecording] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [audioLevel, setAudioLevel] = useState(0);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [liveText, setLiveText] = useState('');

  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const barSeeds = useMemo(() => Array.from({ length: INLINE_BARS }, () => Math.random()), []);
  const processingToastId = useRef<number | null>(null);

  function addToast(message: string, type: 'success' | 'error' | 'loading') {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    if (type !== 'loading') {
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 4000);
    }
    return id;
  }

  function removeToast(id: number) {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }

  // Notify parent when sessionId changes
  useEffect(() => {
    onLiveSessionChange(sessionId);
  }, [sessionId, onLiveSessionChange]);

  // Notify parent when liveText changes
  useEffect(() => {
    onLiveTextChange(liveText);
  }, [liveText, onLiveTextChange]);

  // Auto-scroll to bottom when new segments or live text arrive
  useEffect(() => {
    if (transcriptEndRef.current && !searchQuery) {
      transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [segments, liveText, searchQuery]);

  // Listen for live transcription events
  useEffect(() => {
    const unlistenSegment = listen<Segment>('transcription-segment', (event) => {
      setSegments((prev) => [...prev, event.payload]);
      setLiveText('');
    });

    const unlistenDelta = listen<string>('transcription-delta', (event) => {
      setLiveText((prev) => prev + event.payload);
    });

    const unlistenAudio = listen<number>('audio-level', (event) => {
      setAudioLevel(event.payload);
    });

    const unlistenComplete = listen<string>('session-complete', () => {
      if (processingToastId.current !== null) {
        removeToast(processingToastId.current);
        processingToastId.current = null;
      }
      addToast('Transcription et resume generes avec succes.', 'success');
    });

    const unlistenError = listen<string>('session-error', (event) => {
      if (processingToastId.current !== null) {
        removeToast(processingToastId.current);
        processingToastId.current = null;
      }
      addToast(String(event.payload), 'error');
    });

    return () => {
      unlistenSegment.then((fn) => fn());
      unlistenDelta.then((fn) => fn());
      unlistenAudio.then((fn) => fn());
      unlistenComplete.then((fn) => fn());
      unlistenError.then((fn) => fn());
    };
  }, []);

  // Timer
  useEffect(() => {
    if (!isRecording) return;
    const interval = setInterval(() => {
      setElapsedTime((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isRecording]);

  const handleStart = useCallback(async () => {
    try {
      const id = await invoke<string>('start_session', { mode });
      setSessionId(id);
      setIsRecording(true);
      setSegments([]);
      setElapsedTime(0);
      setSearchQuery('');
    } catch (err) {
      console.error('Erreur au demarrage de la session:', err);
      addToast(String(err), 'error');
    }
  }, [mode]);

  const handleStop = useCallback(async () => {
    if (sessionId) {
      const stoppedId = sessionId;
      try {
        await invoke('stop_session', { sessionId });
        processingToastId.current = addToast('Traitement en cours...', 'loading');
      } catch (err) {
        console.error('Erreur a l\'arret de la session:', err);
      }
      setIsRecording(false);
      setSessionId(null);
      setAudioLevel(0);
      onSessionStopped(stoppedId);
    }
  }, [sessionId, onSessionStopped]);

  const filteredSegments = searchQuery
    ? segments.filter((s) => s.text.toLowerCase().includes(searchQuery.toLowerCase()))
    : segments;

  return (
    <div className="flex flex-col h-full p-6">
      {/* Header — single metadata line */}
      <div className="shrink-0 mb-5">
        <div className="flex flex-wrap items-center gap-2">
          {isRecording ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
              </span>
              <span className="font-mono text-xs text-gray-900 tabular-nums">
                {formatElapsedTime(elapsedTime)}
              </span>
              <span className="text-gray-200">&middot;</span>
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium bg-white/80 text-gray-400 shadow-sm capitalize">
                {mode}
              </span>
              <span className="text-gray-200">&middot;</span>
              {/* Inline audio waveform */}
              <div className="flex items-center gap-[2px] h-3.5">
                {barSeeds.map((seed, i) => {
                  const center = INLINE_BARS / 2;
                  const dist = Math.abs(i - center) / center;
                  const level = Math.max(0, audioLevel * (1 - dist * 0.4) + seed * 0.08);
                  const h = Math.max(0.12, Math.min(1, level));
                  return (
                    <div
                      key={i}
                      className="w-[1.5px] rounded-full bg-gray-400 transition-all duration-150 ease-out"
                      style={{ height: `${h * 100}%`, opacity: 0.4 + h * 0.6 }}
                    />
                  );
                })}
              </div>
            </>
          ) : elapsedTime > 0 ? (
            <>
              <span className="font-mono text-xs text-gray-300 tabular-nums">
                {formatElapsedTime(elapsedTime)}
              </span>
              <span className="text-gray-200">&middot;</span>
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium bg-white/80 text-gray-400 shadow-sm capitalize">
                {mode}
              </span>
            </>
          ) : (
            <>
              <span className="text-xs text-gray-400">En attente</span>
              <span className="text-gray-200">&middot;</span>
              <div className="flex bg-white/80 rounded-full p-0.5 shadow-sm">
                <button
                  onClick={() => setMode('visio')}
                  className={`px-3.5 py-1 text-xs font-medium rounded-full press-scale transition-[background,color,transform] duration-200 ${
                    mode === 'visio'
                      ? 'bg-gray-900 text-white'
                      : 'text-gray-400 hover:text-gray-600'
                  }`}
                >
                  Visio
                </button>
                <button
                  onClick={() => setMode('presentiel')}
                  className={`px-3.5 py-1 text-xs font-medium rounded-full press-scale transition-[background,color,transform] duration-200 ${
                    mode === 'presentiel'
                      ? 'bg-gray-900 text-white'
                      : 'text-gray-400 hover:text-gray-600'
                  }`}
                >
                  Presentiel
                </button>
              </div>
            </>
          )}

          {/* Right side — search + action */}
          <div className="ml-auto flex items-center gap-2">
            {segments.length > 0 && (
              <div className="relative">
                <svg
                  className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-300"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Rechercher..."
                  className="w-44 bg-white/80 rounded-full py-1.5 pl-9 pr-3 text-xs text-gray-900 placeholder-gray-300 focus:outline-none focus:bg-white shadow-input transition-[background,box-shadow] duration-200"
                />
              </div>
            )}

            {isRecording ? (
              <button
                onClick={handleStop}
                className="flex items-center gap-1.5 px-4 py-1.5 bg-red-500 hover:bg-red-600 text-white rounded-full text-xs font-medium press-scale transition-[background,transform] duration-150"
              >
                <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
                Arreter
              </button>
            ) : (
              <button
                onClick={handleStart}
                className="flex items-center gap-1.5 px-4 py-1.5 bg-gray-900 hover:bg-gray-800 text-white rounded-full text-xs font-medium press-scale transition-[background,transform] duration-150"
              >
                <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="6" />
                </svg>
                Demarrer
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Transcript */}
      <div className="flex-1 overflow-auto min-h-0 pr-2">
        {filteredSegments.length === 0 && !liveText ? (
          <div className="flex flex-col items-center justify-center h-full py-20">
            <div className="w-14 h-14 rounded-2xl bg-gray-50 flex items-center justify-center mb-4">
              <svg className="w-7 h-7 text-gray-300" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z" />
              </svg>
            </div>
            <p className="text-sm text-gray-400 text-center max-w-xs">
              {isRecording
                ? 'En attente de la transcription...'
                : segments.length === 0
                  ? 'La transcription apparaitra ici une fois la session demarree.'
                  : 'Aucun resultat pour cette recherche.'}
            </p>
          </div>
        ) : (
          <div className="py-2">
            {filteredSegments.map((segment) => (
              <TranscriptLine
                key={segment.id}
                text={segment.text}
                startTime={segment.start_time}
                highlight={searchQuery || undefined}
              />
            ))}
            {liveText && (() => {
              // Split live text into completed sentences + current fragment
              const parts = liveText.split(/(?<=[.!?…\n])\s+/);
              const completed = parts.slice(0, -1);
              const current = parts[parts.length - 1] || '';
              return (
                <>
                  {completed.map((line, i) => (
                    <div key={`live-${i}`} className="flex gap-2.5 mb-3">
                      <div className="flex flex-col items-start">
                        <div className="px-4 py-2.5 text-sm leading-relaxed bg-gray-50 border border-gray-100 text-gray-400 italic rounded-2xl rounded-tl-md">
                          {line}
                        </div>
                      </div>
                    </div>
                  ))}
                  {current && (
                    <div className="flex gap-2.5 mb-3">
                      <div className="flex flex-col items-start">
                        <div className="px-4 py-2.5 text-sm leading-relaxed bg-gray-50 border border-gray-100 text-gray-400 italic rounded-2xl rounded-tl-md">
                          {current}
                          <span className="inline-block w-1.5 h-4 ml-0.5 bg-gray-300 animate-pulse rounded-sm align-text-bottom" />
                        </div>
                      </div>
                    </div>
                  )}
                </>
              );
            })()}
            <div ref={transcriptEndRef} />
          </div>
        )}
      </div>

      {/* Toast notifications — same style as ExportButtons */}
      {toasts.length > 0 && (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2.5">
          {toasts.map((toast) => (
            <div
              key={toast.id}
              className={`animate-fade-in-up flex items-center gap-2.5 px-5 py-3 rounded-xl text-sm font-medium shadow-lg border ${
                toast.type === 'success'
                  ? 'bg-white border-emerald-100 text-emerald-700'
                  : toast.type === 'error'
                    ? 'bg-white border-red-100 text-red-700'
                    : 'bg-white border-gray-100 text-gray-500'
              }`}
            >
              {toast.type === 'success' ? (
                <svg className="w-4 h-4 text-emerald-400 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                </svg>
              ) : toast.type === 'error' ? (
                <svg className="w-4 h-4 text-red-400 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
                </svg>
              ) : (
                <svg className="w-4 h-4 text-gray-400 shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
              {toast.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

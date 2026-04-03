import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { Session } from '../types';

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  const day = date.getDate();
  const months = [
    'jan.', 'fev.', 'mars', 'avr.', 'mai', 'juin',
    'juil.', 'aout', 'sept.', 'oct.', 'nov.', 'dec.',
  ];
  const month = months[date.getMonth()];
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  return `${day} ${month} ${hours}:${minutes}`;
}

interface SessionListProps {
  refreshKey: number;
  selectedSessionId: string | null;
  onSelectSession: (id: string) => void;
  onSessionDeleted: () => void;
  activeTab: 'session' | 'historique';
  onSwitchToHistorique: () => void;
}

export default function SessionList({
  refreshKey,
  selectedSessionId,
  onSelectSession,
  onSessionDeleted,
  activeTab,
  onSwitchToHistorique,
}: SessionListProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    invoke<Session[]>('get_sessions')
      .then(setSessions)
      .catch(() => setSessions([]))
      .finally(() => setLoading(false));
  }, [refreshKey]);

  const handleClick = (id: string) => {
    onSelectSession(id);
    if (activeTab !== 'historique') {
      onSwitchToHistorique();
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await invoke('delete_session', { sessionId: id });
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (selectedSessionId === id) {
        onSelectSession('');
      }
      onSessionDeleted();
    } catch (err) {
      console.error('Failed to delete session:', err);
    }
  };

  return (
    <aside className="w-56 shrink-0 bg-white rounded-2xl flex flex-col min-h-0 overflow-hidden shadow-panel">
      {/* Header */}
      <div className="px-4 pt-4 pb-3">
        <h2 className="text-[11px] font-medium text-gray-400 uppercase tracking-widest">
          Reunions
        </h2>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-auto px-2 pb-3">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-12">
            <svg className="w-5 h-5 animate-spin text-gray-300" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          </div>
        ) : sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 px-3">
            <p className="text-xs text-gray-300 text-center">
              Aucune reunion enregistree
            </p>
          </div>
        ) : (
          <div className="space-y-0.5">
            {sessions.map((session) => {
              const isSelected = session.id === selectedSessionId;
              return (
                <div
                  key={session.id}
                  onClick={() => handleClick(session.id)}
                  className={`relative w-full text-left px-3 py-2.5 rounded-xl transition-[background,color,transform] duration-200 group cursor-pointer press-scale ${
                    isSelected
                      ? 'bg-gray-100/80 text-gray-900'
                      : 'text-gray-500 hover:bg-gray-50 hover:text-gray-900'
                  }`}
                >
                  <p className={`text-xs truncate pr-5 ${
                    isSelected ? 'font-medium' : 'font-normal'
                  }`}>
                    {session.title}
                  </p>
                  <p className="text-[11px] text-gray-300 mt-0.5">
                    {formatDate(session.created_at)}
                  </p>
                  <button
                    onClick={(e) => handleDelete(e, session.id)}
                    className="absolute right-2 top-2 p-0.5 rounded text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all duration-150"
                    title="Supprimer"
                  >
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}

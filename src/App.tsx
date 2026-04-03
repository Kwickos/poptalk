import { useState, useCallback, useEffect } from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { listen } from '@tauri-apps/api/event';
import { check } from '@tauri-apps/plugin-updater';
import { relaunch } from '@tauri-apps/plugin-process';
import SessionList from './components/SessionList';
import SessionView from './views/SessionView';
import DetailView from './views/DetailView';
import ChatPanel from './components/ChatPanel';
import SettingsModal from './views/SettingsView';

type ActiveTab = 'session' | 'historique';

function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('session');
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  const [liveSessionId, setLiveSessionId] = useState<string | null>(null);
  const [liveText, setLiveText] = useState('');
  const [sessionListRefreshKey, setSessionListRefreshKey] = useState(0);

  // Auto-update
  const [updateAvailable, setUpdateAvailable] = useState<{ version: string; body: string | null } | null>(null);
  const [updating, setUpdating] = useState(false);

  // Listen for native menu "Parametres..." (Cmd+,)
  useEffect(() => {
    const unlisten = listen('open-settings', () => {
      setShowSettings(true);
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  // Refresh session list when background processing completes (title + summary updated)
  useEffect(() => {
    const unlisten = listen('session-complete', () => {
      setSessionListRefreshKey((prev) => prev + 1);
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  // Check for updates on startup
  useEffect(() => {
    check().then((update) => {
      if (update) {
        setUpdateAvailable({ version: update.version, body: update.body ?? null });
      }
    }).catch((e) => {
      console.warn('[updater] Failed to check for updates:', e);
    });
  }, []);

  const handleUpdate = useCallback(async () => {
    setUpdating(true);
    try {
      const update = await check();
      if (update) {
        await update.downloadAndInstall();
        await relaunch();
      }
    } catch (e) {
      console.error('[updater] Update failed:', e);
      setUpdating(false);
    }
  }, []);

  const handleSessionSelect = useCallback((id: string) => {
    setSelectedSessionId(id);
  }, []);

  const handleSessionStopped = useCallback((stoppedSessionId: string) => {
    setSessionListRefreshKey((prev) => prev + 1);
    setActiveTab('historique');
    setSelectedSessionId(stoppedSessionId);
  }, []);

  const handleLiveSessionChange = useCallback((id: string | null) => {
    setLiveSessionId(id);
  }, []);

  const handleLiveTextChange = useCallback((text: string) => {
    setLiveText(text);
  }, []);

  const chatSessionId = activeTab === 'session' ? liveSessionId : selectedSessionId;

  return (
    <div className="flex flex-col h-screen bg-[#eef0f3] text-gray-900">
      {/* Top bar */}
      <header className="shrink-0 h-12 relative">
        {/* Drag layer — covers full header, triggers window drag */}
        <div
          className="absolute inset-0 cursor-default"
          onMouseDown={() => getCurrentWindow().startDragging()}
        />

        {/* Interactive content — pointer-events-none so empty space falls through to drag layer */}
        <div className="relative z-10 flex items-center justify-center h-full px-5 pointer-events-none">
          {/* Center: Tab toggle — floating pill */}
          <div className="flex bg-white/60 backdrop-blur-sm rounded-full p-1 shadow-panel pointer-events-auto">
            <button
              onClick={() => setActiveTab('historique')}
              className={`px-5 py-1.5 rounded-full text-xs font-medium press-scale transition-[background,color,box-shadow] duration-200 ${
                activeTab === 'historique'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-400 hover:text-gray-600'
              }`}
            >
              Historique
            </button>
            <button
              onClick={() => setActiveTab('session')}
              className={`px-5 py-1.5 rounded-full text-xs font-medium press-scale transition-[background,color,box-shadow] duration-200 ${
                activeTab === 'session'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-400 hover:text-gray-600'
              }`}
            >
              Session
            </button>
          </div>
        </div>
      </header>

      {/* Main 3-column layout — floating rounded panels with gaps */}
      <div className="flex flex-1 min-h-0 gap-2 px-2 pb-2">
        {/* Left: Session list — floating panel */}
        <SessionList
          refreshKey={sessionListRefreshKey}
          selectedSessionId={selectedSessionId}
          onSelectSession={handleSessionSelect}
          onSessionDeleted={() => setSelectedSessionId(null)}
          activeTab={activeTab}
          onSwitchToHistorique={() => setActiveTab('historique')}
        />

        {/* Center: Main content — floating panel */}
        <main className="flex-1 min-w-0 bg-white rounded-2xl overflow-hidden shadow-panel">
          {activeTab === 'session' ? (
            <SessionView key="session"
              onSessionStopped={handleSessionStopped}
              onLiveSessionChange={handleLiveSessionChange}
              onLiveTextChange={handleLiveTextChange}
            />
          ) : selectedSessionId ? (
            <DetailView
              sessionId={selectedSessionId}
              onBack={() => setSelectedSessionId(null)}
              onTitleChanged={() => setSessionListRefreshKey((prev) => prev + 1)}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="w-12 h-12 rounded-2xl bg-gray-50 flex items-center justify-center mb-3">
                <svg className="w-6 h-6 text-gray-300" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                </svg>
              </div>
              <p className="text-xs text-gray-400 text-center max-w-xs">
                Selectionnez une reunion dans la liste pour voir sa transcription.
              </p>
            </div>
          )}
        </main>

        {/* Right: Chat panel — floating panel */}
        <ChatPanel sessionId={chatSessionId} liveText={activeTab === 'session' ? liveText : ''} />
      </div>

      {/* Settings modal */}
      {showSettings && (
        <SettingsModal onClose={() => setShowSettings(false)} />
      )}

      {/* Update banner */}
      {updateAvailable && (
        <div className="fixed bottom-4 right-4 bg-white rounded-xl shadow-panel p-4 max-w-xs z-50 animate-fade-in-up">
          <p className="text-sm font-medium text-gray-900 mb-1">
            Mise a jour {updateAvailable.version}
          </p>
          <p className="text-xs text-gray-500 mb-3">
            Une nouvelle version est disponible.
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setUpdateAvailable(null)}
              className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
            >
              Plus tard
            </button>
            <button
              onClick={handleUpdate}
              disabled={updating}
              className="px-3 py-1.5 text-xs bg-gray-900 text-white rounded-lg hover:bg-gray-800 press-scale transition-[background,transform] duration-150 disabled:opacity-50"
            >
              {updating ? 'Mise a jour...' : 'Installer'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

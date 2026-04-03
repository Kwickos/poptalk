import { useState, useEffect, useCallback, type ReactNode } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { check } from '@tauri-apps/plugin-updater';
import { relaunch } from '@tauri-apps/plugin-process';
import { getVersion } from '@tauri-apps/api/app';

interface AudioDevice {
  name: string;
  is_default: boolean;
}

interface ModelInfo {
  id: string;
  name: string;
  size_bytes: number;
  url: string;
  filename: string;
}

interface ModelStatus {
  status: 'not_downloaded' | 'downloading' | 'ready';
  progress?: number;
  path?: string;
}

interface SettingsModalProps {
  onClose: () => void;
}

type Category = 'models' | 'audio' | 'ia' | 'export' | 'about';

const categories: { id: Category; label: string; icon: ReactNode }[] = [
  {
    id: 'models',
    label: 'Modeles',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
      </svg>
    ),
  },
  {
    id: 'audio',
    label: 'Audio',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
      </svg>
    ),
  },
  {
    id: 'ia',
    label: 'Intelligence IA',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
      </svg>
    ),
  },
  {
    id: 'export',
    label: 'Export',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
    ),
  },
  {
    id: 'about' as Category,
    label: 'A propos',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
      </svg>
    ),
  },
];

function formatSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} Go`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} Mo`;
  return `${(bytes / 1_000).toFixed(0)} Ko`;
}

export default function SettingsModal({ onClose }: SettingsModalProps) {
  const [activeCategory, setActiveCategory] = useState<Category>('models');
  const [apiKey, setApiKey] = useState('');
  const [inputDevice, setInputDevice] = useState('');
  const [exportDir, setExportDir] = useState('');
  const [whisperModel, setWhisperModel] = useState('large-v3-turbo-q5_0');
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelStatuses, setModelStatuses] = useState<Record<string, ModelStatus>>({});
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
  const [updateStatus, setUpdateStatus] = useState<'idle' | 'checking' | 'available' | 'up-to-date' | 'downloading' | 'error'>('idle');
  const [updateVersion, setUpdateVersion] = useState('');
  const [appVersion, setAppVersion] = useState('');

  useEffect(() => {
    getVersion().then(setAppVersion).catch(() => setAppVersion('?'));
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const [key, deviceSetting, exportSetting, whisperSetting, deviceList, modelList] = await Promise.all([
          invoke<string>('get_api_key'),
          invoke<string | null>('get_setting', { key: 'input_device' }),
          invoke<string | null>('get_setting', { key: 'export_dir' }),
          invoke<string | null>('get_setting', { key: 'whisper_model' }),
          invoke<AudioDevice[]>('list_input_devices'),
          invoke<ModelInfo[]>('get_available_models'),
        ]);
        setApiKey(key);
        setInputDevice(deviceSetting ?? '');
        setExportDir(exportSetting ?? '');
        setWhisperModel(whisperSetting ?? 'large-v3-turbo-q5_0');
        setDevices(deviceList);
        setModels(modelList);

        const statuses: Record<string, ModelStatus> = {};
        for (const model of modelList) {
          const status = await invoke<ModelStatus>('get_model_status', { modelId: model.id });
          statuses[model.id] = status;
        }
        setModelStatuses(statuses);
      } catch (err) {
        console.error('Erreur chargement parametres:', err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    const unlisten = listen<{ model_id: string; progress: number }>('model-download-progress', (event) => {
      setDownloadProgress(event.payload.progress);
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  useEffect(() => {
    if (!feedback) return;
    const timer = setTimeout(() => setFeedback(null), 3000);
    return () => clearTimeout(timer);
  }, [feedback]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setFeedback(null);
    try {
      await Promise.all([
        invoke('set_api_key', { key: apiKey }),
        invoke('set_setting', { key: 'input_device', value: inputDevice }),
        invoke('set_setting', { key: 'export_dir', value: exportDir }),
        invoke('set_setting', { key: 'whisper_model', value: whisperModel }),
      ]);
      setFeedback({ type: 'success', message: 'Parametres sauvegardes avec succes.' });
    } catch (err) {
      console.error('Erreur sauvegarde parametres:', err);
      setFeedback({ type: 'error', message: `Erreur lors de la sauvegarde: ${err}` });
    } finally {
      setSaving(false);
    }
  }, [apiKey, inputDevice, exportDir, whisperModel]);

  const handleDownloadModel = useCallback(async (modelId: string) => {
    setDownloadingModel(modelId);
    setDownloadProgress(0);
    try {
      await invoke('download_model', { modelId });
      const status = await invoke<ModelStatus>('get_model_status', { modelId });
      setModelStatuses(prev => ({ ...prev, [modelId]: status }));
      setFeedback({ type: 'success', message: 'Modele telecharge avec succes.' });
    } catch (err) {
      console.error('Erreur telechargement modele:', err);
      setFeedback({ type: 'error', message: `Erreur telechargement: ${err}` });
    } finally {
      setDownloadingModel(null);
    }
  }, []);

  const handlePickFolder = useCallback(async () => {
    try {
      const folder = await invoke<string | null>('pick_folder');
      if (folder) setExportDir(folder);
    } catch (err) {
      console.error('Erreur selection dossier:', err);
    }
  }, []);

  const maskedValue = apiKey.length > 4 ? apiKey.slice(0, 4) + '\u2022'.repeat(Math.min(apiKey.length - 4, 32)) : apiKey;

  const whisperModels = models.filter(m => !m.id.startsWith('diarize-'));
  const diarizeModels = models.filter(m => m.id.startsWith('diarize-'));

  // ── Model download row ──
  const ModelRow = ({ model }: { model: ModelInfo }) => {
    const status = modelStatuses[model.id];
    const isReady = status?.status === 'ready';
    const isDownloading = downloadingModel === model.id;

    return (
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50/80 rounded-xl border border-gray-100/60">
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium text-gray-800 truncate block">{model.name}</span>
          <span className="text-xs text-gray-400 mt-0.5 block">{formatSize(model.size_bytes)}</span>
        </div>
        {isReady ? (
          <span className="text-xs text-emerald-600 font-medium px-2.5 py-1 bg-emerald-50 border border-emerald-100 rounded-lg shrink-0">Pret</span>
        ) : isDownloading ? (
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-28 h-1.5 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-gray-800 rounded-full transition-all duration-300"
                style={{ width: `${Math.round(downloadProgress * 100)}%` }}
              />
            </div>
            <span className="text-xs text-gray-500 w-10 text-right tabular-nums">{Math.round(downloadProgress * 100)}%</span>
          </div>
        ) : (
          <button
            onClick={() => handleDownloadModel(model.id)}
            disabled={downloadingModel !== null}
            className="text-xs text-gray-700 font-medium px-3.5 py-1.5 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-gray-300 transition-all disabled:opacity-40 shrink-0"
          >
            Telecharger
          </button>
        )}
      </div>
    );
  };

  // ── Section renderers ──

  const renderModels = () => (
    <div className="space-y-6">
      {/* Transcription models */}
      <div>
        <h3 className="text-sm font-semibold text-gray-900 mb-1">Transcription (Whisper)</h3>
        <p className="text-xs text-gray-400 mb-3">Modele utilise pour convertir la parole en texte. Le modele Small est recommande pour les machines sans GPU.</p>

        <select
          value={whisperModel}
          onChange={(e) => setWhisperModel(e.target.value)}
          className="w-full px-4 py-2.5 mb-3 bg-gray-50 border border-gray-100 rounded-xl text-sm text-gray-900 focus:outline-none focus:bg-white focus:border-gray-200 focus:ring-0 transition-all duration-150 appearance-none"
        >
          {whisperModels.map(model => (
            <option key={model.id} value={model.id}>
              {model.name} ({formatSize(model.size_bytes)})
            </option>
          ))}
        </select>

        <div className="space-y-2">
          {whisperModels.map(model => <ModelRow key={model.id} model={model} />)}
        </div>
      </div>

      {/* Diarization models */}
      <div>
        <h3 className="text-sm font-semibold text-gray-900 mb-1">Diarisation (identification des speakers)</h3>
        <p className="text-xs text-gray-400 mb-3">Ces deux modeles travaillent ensemble pour identifier qui parle. Telechargez les deux pour activer la diarisation.</p>

        <div className="space-y-2">
          {diarizeModels.map(model => <ModelRow key={model.id} model={model} />)}
        </div>
      </div>
    </div>
  );

  const renderAudio = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-semibold text-gray-900 mb-1">Microphone d'entree</h3>
        <p className="text-xs text-gray-400 mb-3">Peripherique audio utilise pour l'enregistrement des sessions.</p>
        <select
          id="input-device"
          value={inputDevice}
          onChange={(e) => setInputDevice(e.target.value)}
          className="w-full px-4 py-2.5 bg-gray-50 border border-gray-100 rounded-xl text-sm text-gray-900 focus:outline-none focus:bg-white focus:border-gray-200 focus:ring-0 transition-all duration-150 appearance-none"
        >
          <option value="">Par defaut (systeme)</option>
          {devices.map((device) => (
            <option key={device.name} value={device.name}>
              {device.name}{device.is_default ? ' (defaut)' : ''}
            </option>
          ))}
        </select>
      </div>
    </div>
  );

  const renderIA = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-semibold text-gray-900 mb-1">
          Cle API LLM
          <span className="ml-2 text-xs font-normal text-gray-400">optionnel</span>
        </h3>
        <p className="text-xs text-gray-400 mb-3">
          Necessaire uniquement pour la generation de resumes et la recherche IA dans les transcriptions. Compatible Mistral, OpenAI, ou tout endpoint OpenAI-compatible.
        </p>
        <div className="relative">
          <input
            id="api-key"
            type="text"
            value={showKey ? apiKey : maskedValue}
            onChange={(e) => {
              setShowKey(true);
              setApiKey(e.target.value);
            }}
            onFocus={() => setShowKey(true)}
            placeholder="sk-..."
            className="w-full px-4 py-2.5 pr-12 bg-gray-50 border border-gray-100 rounded-xl text-sm text-gray-900 placeholder-gray-300 focus:outline-none focus:bg-white focus:border-gray-200 focus:ring-0 transition-all duration-150 font-mono"
          />
          <button
            type="button"
            onClick={() => setShowKey(!showKey)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-300 hover:text-gray-500 transition-colors p-1"
            title={showKey ? 'Masquer la cle' : 'Afficher la cle'}
          >
            {showKey ? (
              <svg className="w-4.5 h-4.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L6.59 6.59m7.532 7.532l3.29 3.29M3 3l18 18" />
              </svg>
            ) : (
              <svg className="w-4.5 h-4.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            )}
          </button>
        </div>

        <div className="mt-4 px-4 py-3 bg-amber-50/60 border border-amber-100 rounded-xl">
          <p className="text-xs text-amber-700">
            Sans cle API, la transcription fonctionne normalement. Seuls les resumes automatiques et la recherche IA seront desactives.
          </p>
        </div>
      </div>
    </div>
  );

  const renderExport = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-semibold text-gray-900 mb-1">Dossier d'export</h3>
        <p className="text-xs text-gray-400 mb-3">Emplacement ou les fichiers Markdown et PDF exportes seront sauvegardes.</p>
        <div className="flex items-center gap-2">
          <div className="flex-1 px-4 py-2.5 bg-gray-50 border border-gray-100 rounded-xl text-sm text-gray-500 truncate">
            {exportDir || '~/Documents/poptalk/exports/ (par defaut)'}
          </div>
          <button
            type="button"
            onClick={handlePickFolder}
            className="shrink-0 px-4 py-2.5 bg-gray-50 border border-gray-100 rounded-xl text-sm text-gray-700 hover:bg-gray-100 transition-all duration-150"
          >
            Parcourir...
          </button>
          {exportDir && (
            <button
              type="button"
              onClick={() => setExportDir('')}
              className="shrink-0 p-2.5 bg-gray-50 border border-gray-100 rounded-xl text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-all duration-150"
              title="Reinitialiser"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );

  const handleCheckUpdate = useCallback(async () => {
    setUpdateStatus('checking');
    try {
      const update = await check();
      if (update?.available) {
        setUpdateVersion(update.version);
        setUpdateStatus('available');
      } else {
        setUpdateStatus('up-to-date');
      }
    } catch (e: any) {
      console.error('[updater] Check failed:', e);
      // If the error is about no update available, treat as up-to-date
      const msg = String(e?.message || e || '');
      if (msg.includes('up to date') || msg.includes('UpToDate')) {
        setUpdateStatus('up-to-date');
      } else {
        setUpdateStatus('error');
      }
    }
  }, []);

  const handleInstallUpdate = useCallback(async () => {
    setUpdateStatus('downloading');
    try {
      const update = await check();
      if (update) {
        await update.downloadAndInstall();
        await relaunch();
      }
    } catch (e) {
      console.error('[updater] Install failed:', e);
      setUpdateStatus('error');
    }
  }, []);

  const renderAbout = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-semibold text-gray-900 mb-1">PopTalk</h3>
        <p className="text-xs text-gray-400 mb-4">
          Version {appVersion} — Transcription locale de reunions avec diarisation.
        </p>

        <div className="space-y-3">
          <button
            onClick={handleCheckUpdate}
            disabled={updateStatus === 'checking' || updateStatus === 'downloading'}
            className="flex items-center gap-2.5 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm text-gray-700 hover:bg-gray-100 transition-all disabled:opacity-50"
          >
            {updateStatus === 'checking' ? (
              <>
                <svg className="w-4 h-4 animate-spin text-gray-400" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Verification...
              </>
            ) : updateStatus === 'downloading' ? (
              <>
                <svg className="w-4 h-4 animate-spin text-gray-400" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Telechargement...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
                </svg>
                Verifier les mises a jour
              </>
            )}
          </button>

          {updateStatus === 'available' && (
            <div className="flex items-center justify-between px-4 py-3 bg-blue-50 border border-blue-100 rounded-xl">
              <div>
                <p className="text-sm font-medium text-blue-900">Version {updateVersion} disponible</p>
                <p className="text-xs text-blue-600 mt-0.5">Une nouvelle version est prete a etre installee.</p>
              </div>
              <button
                onClick={handleInstallUpdate}
                className="shrink-0 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded-lg transition-all"
              >
                Installer
              </button>
            </div>
          )}

          {updateStatus === 'up-to-date' && (
            <div className="px-4 py-3 bg-emerald-50 border border-emerald-100 rounded-xl">
              <p className="text-sm text-emerald-700">Vous etes a jour.</p>
            </div>
          )}

          {updateStatus === 'error' && (
            <div className="px-4 py-3 bg-red-50 border border-red-100 rounded-xl">
              <p className="text-sm text-red-700">Impossible de verifier les mises a jour. Verifiez votre connexion.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeCategory) {
      case 'models': return renderModels();
      case 'audio': return renderAudio();
      case 'ia': return renderIA();
      case 'export': return renderExport();
      case 'about': return renderAbout();
    }
  };

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/20 z-50" onClick={onClose} />

      {/* Modal */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
        <div
          className="bg-white rounded-2xl shadow-panel w-full max-w-2xl pointer-events-auto animate-scale-pop flex flex-col"
          style={{ maxHeight: '80vh' }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 shrink-0">
            <h2 className="text-base font-semibold text-gray-900 tracking-tight">Parametres</h2>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-50 transition-all duration-150"
            >
              <svg className="w-4.5 h-4.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Body: sidebar + content */}
          <div className="flex flex-1 min-h-0">
            {/* Sidebar */}
            <nav className="w-44 shrink-0 border-r border-gray-100 py-3 px-2 space-y-0.5">
              {categories.map(cat => (
                <button
                  key={cat.id}
                  onClick={() => setActiveCategory(cat.id)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left text-sm transition-all duration-150 ${
                    activeCategory === cat.id
                      ? 'bg-gray-100 text-gray-900 font-medium'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <span className={activeCategory === cat.id ? 'text-gray-700' : 'text-gray-400'}>{cat.icon}</span>
                  {cat.label}
                </button>
              ))}
            </nav>

            {/* Content */}
            <div className="flex-1 min-w-0 flex flex-col">
              <div className="flex-1 overflow-y-auto px-6 py-5">
                {loading ? (
                  <div className="flex items-center gap-3 py-8 justify-center">
                    <svg className="w-5 h-5 animate-spin text-gray-300" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    <span className="text-sm text-gray-400">Chargement...</span>
                  </div>
                ) : (
                  renderContent()
                )}
              </div>

              {/* Footer: save + feedback */}
              <div className="shrink-0 px-6 py-3 border-t border-gray-100 flex items-center gap-3">
                <button
                  onClick={handleSave}
                  disabled={loading || saving}
                  className={`flex items-center gap-2 px-5 py-2 bg-gray-900 hover:bg-gray-800 text-white rounded-lg text-sm font-medium press-scale transition-[background,transform] duration-150 shadow-sm ${
                    loading || saving ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  {saving ? (
                    <>
                      <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Sauvegarde...
                    </>
                  ) : (
                    'Sauvegarder'
                  )}
                </button>

                {feedback && (
                  <div className={`animate-fade-in-down flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs ${
                    feedback.type === 'success'
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-100'
                      : 'bg-red-50 text-red-700 border border-red-100'
                  }`}>
                    {feedback.type === 'success' ? (
                      <svg className="w-3.5 h-3.5 text-emerald-400 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                      </svg>
                    ) : (
                      <svg className="w-3.5 h-3.5 text-red-400 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
                      </svg>
                    )}
                    {feedback.message}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

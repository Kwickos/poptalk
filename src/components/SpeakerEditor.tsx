import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

interface SpeakerEditorProps {
  sessionId: string;
  currentName: string;
  onRenamed: (oldName: string, newName: string) => void;
  onClose: () => void;
}

export default function SpeakerEditor({
  sessionId,
  currentName,
  onRenamed,
  onClose,
}: SpeakerEditorProps) {
  const [newName, setNewName] = useState(currentName);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRename() {
    const trimmed = newName.trim();
    if (!trimmed || trimmed === currentName) {
      onClose();
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await invoke('rename_speaker', {
        sessionId,
        oldName: currentName,
        newName: trimmed,
      });
      onRenamed(currentName, trimmed);
      onClose();
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') {
      handleRename();
    } else if (e.key === 'Escape') {
      onClose();
    }
  }

  return (
    <div className="absolute z-50 top-full left-0 mt-2 bg-white rounded-2xl shadow-panel p-4 min-w-[260px] animate-scale-pop">
      <label className="block text-xs font-medium text-gray-500 mb-2">
        Renommer le locuteur
      </label>
      <input
        type="text"
        value={newName}
        onChange={(e) => setNewName(e.target.value)}
        onKeyDown={handleKeyDown}
        autoFocus
        className="w-full px-3.5 py-2.5 bg-gray-50 rounded-xl text-sm text-gray-900 placeholder-gray-300 focus:outline-none focus:bg-white shadow-input transition-[background,box-shadow] duration-200"
        placeholder="Nom du locuteur"
      />
      {error && (
        <p className="text-xs text-red-500 mt-2">{error}</p>
      )}
      <div className="flex gap-2 mt-3">
        <button
          onClick={handleRename}
          disabled={loading}
          className="flex-1 px-4 py-2 bg-gray-900 hover:bg-gray-800 disabled:opacity-50 text-white text-xs font-medium rounded-lg press-scale transition-[background,transform] duration-150"
        >
          {loading ? 'En cours...' : 'Renommer'}
        </button>
        <button
          onClick={onClose}
          disabled={loading}
          className="flex-1 px-4 py-2 bg-gray-50 hover:bg-gray-100 disabled:opacity-50 text-gray-600 text-xs font-medium rounded-lg transition-all duration-150"
        >
          Annuler
        </button>
      </div>
    </div>
  );
}

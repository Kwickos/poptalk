import { Facehash } from 'facehash';

const AVATAR_COLORS = [
  '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e',
  '#f97316', '#eab308', '#22c55e', '#14b8a6',
  '#06b6d4', '#3b82f6', '#a855f7', '#e11d48',
];

interface TranscriptLineProps {
  text: string;
  startTime: number;
  highlight?: string;
  speaker?: string | null;
  isUser?: boolean;
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function highlightText(text: string, query: string): React.ReactNode {
  if (!query) return text;

  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const parts = text.split(regex);

  return parts.map((part, i) =>
    regex.test(part) ? (
      <mark key={i} className="bg-amber-100/70 text-amber-900 rounded px-0.5">
        {part}
      </mark>
    ) : (
      part
    )
  );
}

export default function TranscriptLine({ text, startTime, highlight, speaker, isUser }: TranscriptLineProps) {
  // During live recording (no speaker info), show as neutral left-aligned bubble
  const hasNoSpeaker = speaker === undefined || speaker === null;
  const alignRight = !hasNoSpeaker && isUser;

  return (
    <div className={`flex gap-2.5 mb-3 animate-segment-enter ${alignRight ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      {!hasNoSpeaker && (
        <div className="shrink-0 mt-5">
          <Facehash name={speaker || 'Inconnu'} size={28} colors={AVATAR_COLORS} style={{ borderRadius: '25%' }} />
        </div>
      )}

      {/* Content */}
      <div className={`flex flex-col max-w-[75%] ${alignRight ? 'items-end' : 'items-start'}`}>
        {/* Speaker name and timestamp */}
        <div className={`flex items-center gap-2 mb-1 ${alignRight ? 'flex-row-reverse' : ''}`}>
          {!hasNoSpeaker && (
            <span className="text-xs font-medium text-gray-500">
              {speaker}
            </span>
          )}
          <span className="text-[10px] text-gray-300 font-mono tabular-nums">
            {formatTimestamp(startTime)}
          </span>
        </div>

        {/* Chat bubble */}
        <div
          className={`px-4 py-2.5 text-sm leading-relaxed ${
            alignRight
              ? 'bg-gray-900 text-white rounded-2xl rounded-tr-md'
              : 'bg-gray-50 border border-gray-100 text-gray-700 rounded-2xl rounded-tl-md'
          }`}
        >
          {highlight ? highlightText(text, highlight) : text}
        </div>
      </div>
    </div>
  );
}

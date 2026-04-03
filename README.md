# PopTalk

Application de transcription de reunions en temps reel, avec diarisation des speakers et resume automatique par IA. Disponible sur **macOS** et **Windows**.

## Fonctionnalites

- **Transcription locale en temps reel** — Audio capture et transcrit par Whisper (whisper.cpp) directement sur votre machine
- **Diarisation** — Identification automatique des speakers via sherpa-onnx avec avatars uniques (facehash)
- **Resume IA** — Generation automatique d'un resume structure (optionnel, via API LLM)
- **Chat IA** — Posez des questions sur la transcription en cours ou passee
- **Export** — Markdown, PDF
- **Recherche** — Recherche plein texte dans les transcriptions (FTS5)
- **Modes** — Visio (audio systeme) et Presentiel (microphone)
- **100% offline** — Transcription et diarisation fonctionnent sans connexion internet

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Desktop | [Tauri 2](https://v2.tauri.app/) |
| Backend | Rust (tokio, reqwest, rusqlite, cpal) |
| Frontend | React 19, TypeScript, Tailwind CSS 4 |
| Transcription | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via whisper-rs (local) |
| Diarisation | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) via sherpa-rs (local) |
| Resume / Chat | API LLM OpenAI-compatible (Mistral, OpenAI, etc.) — optionnel |
| Base de donnees | SQLite (embarquee, FTS5) |
| Audio systeme | ScreenCaptureKit (macOS) / WASAPI loopback (Windows) |

## Telechargement

Telecharger le dernier installeur depuis la [page Releases](https://github.com/Kwickos/poptalk/releases) :

| Plateforme | Format |
|------------|--------|
| macOS | `.dmg` |
| Windows | `.msi` / `.exe` |

## Pre-requis (build depuis les sources)

- macOS 13+ ou Windows 10+
- [Rust](https://rustup.rs/) (stable)
- [Node.js](https://nodejs.org/) 18+

## Installation depuis les sources

```bash
git clone https://github.com/Kwickos/poptalk.git
cd poptalk
npm install
npm run tauri dev
```

Pour builder l'installeur :

```bash
npm run tauri build
```

Les artefacts seront dans `src-tauri/target/release/bundle/`.

## Configuration

Au premier lancement, aller dans **Parametres > Modeles** et telecharger les modeles necessaires :
- **Whisper** (transcription) — obligatoire
- **Segmentation + Embedding** (diarisation) — recommande

Une cle API LLM est optionnelle (uniquement pour les resumes et la recherche IA).

## Architecture

```
poptalk/
├── src/                    # Frontend React
│   ├── components/         # Composants UI (ChatPanel, TranscriptLine, etc.)
│   ├── views/              # Vues principales (SessionView, DetailView, etc.)
│   └── styles.css          # Styles globaux + animations
├── src-tauri/              # Backend Rust
│   └── src/
│       ├── audio/          # Capture audio (systeme + micro)
│       ├── whisper/        # Transcription locale (whisper.cpp)
│       ├── diarization/    # Diarisation speakers (sherpa-onnx)
│       ├── models/         # Gestion telechargement modeles
│       ├── llm/            # Client API LLM (resumes, recherche)
│       ├── db/             # SQLite + FTS5
│       └── commands.rs     # Commandes Tauri
├── .github/workflows/      # CI/CD (build macOS + Windows)
└── package.json
```

## Licence

Projet prive.

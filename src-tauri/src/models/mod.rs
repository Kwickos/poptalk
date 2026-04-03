use std::path::{Path, PathBuf};
use serde::Serialize;
use tokio::sync::mpsc;

/// Describes a downloadable model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub size_bytes: u64,
    pub url: String,
    pub filename: String,
}

/// Status of a model on disk.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status")]
pub enum ModelStatus {
    #[serde(rename = "not_downloaded")]
    NotDownloaded,
    #[serde(rename = "downloading")]
    Downloading { progress: f64 },
    #[serde(rename = "ready")]
    Ready { path: String },
}

/// Returns the base directory for stored models.
pub fn models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("poptalk")
        .join("models")
}

// ── Whisper models ──────────────────────────────────────────────────

pub fn whisper_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "large-v3-turbo-q5_0".into(),
            name: "Large V3 Turbo quantifie (recommande)".into(),
            size_bytes: 547_000_000,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin".into(),
            filename: "ggml-large-v3-turbo-q5_0.bin".into(),
        },
        ModelInfo {
            id: "medium-q5_0".into(),
            name: "Medium quantifie (bon compromis)".into(),
            size_bytes: 514_000_000,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin".into(),
            filename: "ggml-medium-q5_0.bin".into(),
        },
        ModelInfo {
            id: "small".into(),
            name: "Small (leger, rapide)".into(),
            size_bytes: 488_000_000,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin".into(),
            filename: "ggml-small.bin".into(),
        },
        ModelInfo {
            id: "medium".into(),
            name: "Medium (bonne qualite, lent sur CPU)".into(),
            size_bytes: 1_530_000_000,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin".into(),
            filename: "ggml-medium.bin".into(),
        },
        ModelInfo {
            id: "large-v3-turbo".into(),
            name: "Large V3 Turbo (meilleure qualite, GPU recommande)".into(),
            size_bytes: 1_620_000_000,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin".into(),
            filename: "ggml-large-v3-turbo.bin".into(),
        },
    ]
}

// ── Diarization models ──────────────────────────────────────────────

pub fn diarization_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "diarize-segmentation".into(),
            name: "Segmentation Reverb v1 (recommande)".into(),
            size_bytes: 10_900_000,
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v1.tar.bz2".into(),
            filename: "sherpa-onnx-reverb-diarization-v1".into(),
        },
        ModelInfo {
            id: "diarize-embedding".into(),
            name: "Speaker Embedding CAM++ LM (recommande)".into(),
            size_bytes: 29_300_000,
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++_LM.onnx".into(),
            filename: "wespeaker_en_voxceleb_CAM++_LM.onnx".into(),
        },
    ]
}

/// Get all available models.
pub fn all_models() -> Vec<ModelInfo> {
    let mut models = whisper_models();
    models.extend(diarization_models());
    models
}

/// Find model info by ID.
pub fn find_model(model_id: &str) -> Option<ModelInfo> {
    all_models().into_iter().find(|m| m.id == model_id)
}

/// Get the status of a model (downloaded or not).
pub fn get_model_status(model_id: &str) -> ModelStatus {
    if let Some(model) = find_model(model_id) {
        let path = models_dir().join(&model.filename);
        if path.exists() {
            ModelStatus::Ready {
                path: path.to_string_lossy().to_string(),
            }
        } else {
            ModelStatus::NotDownloaded
        }
    } else {
        ModelStatus::NotDownloaded
    }
}

/// Get the path to a downloaded model, or None if not downloaded.
pub fn get_model_path(model_id: &str) -> Option<PathBuf> {
    let model = find_model(model_id)?;
    let path = models_dir().join(&model.filename);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Get path to the configured whisper model (from DB setting or default).
pub fn get_whisper_model_path(whisper_model_id: &str) -> Option<PathBuf> {
    get_model_path(whisper_model_id)
}

/// Get paths to diarization models.
pub fn get_diarization_paths() -> Option<(PathBuf, PathBuf)> {
    let seg = get_model_path("diarize-segmentation")?;
    let emb = get_model_path("diarize-embedding")?;
    // The segmentation model is in a tar directory, find the .onnx file
    let seg_onnx = seg.join("model.onnx");
    if seg_onnx.exists() {
        Some((seg_onnx, emb))
    } else {
        // Try to find any .onnx in the directory
        if seg.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&seg) {
                for entry in entries.flatten() {
                    if entry.path().extension().map_or(false, |e| e == "onnx") {
                        return Some((entry.path(), emb));
                    }
                }
            }
        }
        // If it's a direct .onnx file
        if seg.extension().map_or(false, |e| e == "onnx") {
            Some((seg, emb))
        } else {
            None
        }
    }
}

/// Download a model with progress reporting.
/// Sends progress as f64 (0.0 to 1.0) through the channel.
pub async fn download_model(
    model_id: &str,
    progress_tx: mpsc::UnboundedSender<f64>,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let model = find_model(model_id)
        .ok_or_else(|| format!("Modele inconnu: {}", model_id))?;

    let dir = models_dir();
    std::fs::create_dir_all(&dir)?;

    let dest = dir.join(&model.filename);

    // If it's a tar.bz2, download to a temp file then extract
    if model.url.ends_with(".tar.bz2") {
        let temp_file = dir.join(format!("{}.tar.bz2", model.id));
        download_file(&model.url, &temp_file, model.size_bytes, &progress_tx).await?;
        extract_tar_bz2(&temp_file, &dir)?;
        std::fs::remove_file(&temp_file).ok();
    } else {
        download_file(&model.url, &dest, model.size_bytes, &progress_tx).await?;
    }

    let _ = progress_tx.send(1.0);
    Ok(dest)
}

async fn download_file(
    url: &str,
    dest: &Path,
    expected_size: u64,
    progress_tx: &mpsc::UnboundedSender<f64>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::AsyncWriteExt;

    let client = reqwest::Client::new();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(format!("Erreur telechargement: HTTP {}", response.status()).into());
    }

    let total = response.content_length().unwrap_or(expected_size);
    let mut downloaded: u64 = 0;

    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = response.bytes_stream();

    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        let progress = (downloaded as f64 / total as f64).min(1.0);
        let _ = progress_tx.send(progress);
    }

    file.flush().await?;
    Ok(())
}

fn extract_tar_bz2(
    archive_path: &Path,
    dest_dir: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::process::Command;
    let status = Command::new("tar")
        .arg("-xjf")
        .arg(archive_path)
        .arg("-C")
        .arg(dest_dir)
        .status()?;
    if !status.success() {
        return Err("Erreur extraction archive tar.bz2".into());
    }
    Ok(())
}

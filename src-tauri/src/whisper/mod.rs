pub mod realtime;

use std::path::Path;

/// A transcription segment with timing info.
#[derive(Debug, Clone)]
pub struct WhisperSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// Result of a full file transcription.
#[derive(Debug, Clone)]
pub struct TranscriptionResponse {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
}

#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub speaker_id: Option<String>,
}

/// Number of threads to use for whisper inference.
/// i5 quad-core: use all 4 cores.
const N_THREADS: i32 = 4;

/// Create optimized FullParams for French transcription.
fn make_params() -> whisper_rs::FullParams<'static, 'static> {
    let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(N_THREADS);
    params.set_language(Some("fr"));
    params.set_translate(false);
    params.set_no_timestamps(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_single_segment(false);
    params.set_token_timestamps(false);
    // Speed optimizations
    params.set_no_context(true); // don't use previous context (faster for chunks)
    params
}

/// Transcribe a chunk of PCM audio (i16, 16kHz mono) using a reusable state.
/// Returns segments with timestamps offset by `offset_secs`.
pub fn transcribe_chunk_with_state(
    state: &mut whisper_rs::WhisperState,
    samples_i16: &[i16],
    offset_secs: f64,
) -> Result<Vec<WhisperSegment>, Box<dyn std::error::Error + Send + Sync>> {
    // Convert i16 to f32 as whisper.cpp expects
    let samples_f32: Vec<f32> = samples_i16.iter().map(|&s| s as f32 / 32768.0).collect();

    let params = make_params();
    state.full(params, &samples_f32).map_err(|e| format!("Erreur inference Whisper: {:?}", e))?;

    let num_segments = state.full_n_segments();
    let mut segments = Vec::new();

    for i in 0..num_segments {
        if let Some(seg) = state.get_segment(i) {
            let text = seg.to_str_lossy().map_err(|e| format!("Erreur texte segment: {:?}", e))?;
            let start = seg.start_timestamp() as f64 / 100.0;
            let end = seg.end_timestamp() as f64 / 100.0;

            let trimmed = text.trim();
            if !trimmed.is_empty() {
                segments.push(WhisperSegment {
                    text: trimmed.to_string(),
                    start: start + offset_secs,
                    end: end + offset_secs,
                });
            }
        }
    }

    Ok(segments)
}

/// Transcribe a full WAV file locally using whisper.cpp.
/// This produces higher quality results than chunked real-time transcription.
pub fn transcribe_file(
    model_path: &Path,
    audio_path: &Path,
) -> Result<TranscriptionResponse, Box<dyn std::error::Error + Send + Sync>> {
    // Load audio file
    let reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let samples_i16: Vec<i16> = if spec.sample_format == hound::SampleFormat::Float {
        reader.into_samples::<f32>()
            .filter_map(|s| s.ok())
            .map(|s| (s * 32767.0) as i16)
            .collect()
    } else {
        reader.into_samples::<i16>()
            .filter_map(|s| s.ok())
            .collect()
    };

    // Resample to 16kHz if needed
    let samples_16k = if spec.sample_rate != 16000 {
        resample(&samples_i16, spec.sample_rate, 16000)
    } else {
        samples_i16
    };

    // Downmix to mono if needed
    let samples_mono = if spec.channels > 1 {
        downmix_to_mono(&samples_16k, spec.channels as usize)
    } else {
        samples_16k
    };

    // Convert to f32
    let samples_f32: Vec<f32> = samples_mono.iter().map(|&s| s as f32 / 32768.0).collect();

    // Load model
    let ctx = whisper_rs::WhisperContext::new_with_params(
        model_path.to_str().unwrap_or(""),
        whisper_rs::WhisperContextParameters::default(),
    )
    .map_err(|e| format!("Erreur chargement modele Whisper: {:?}", e))?;

    let mut params = make_params();
    // For batch processing, allow context (better quality)
    params.set_no_context(false);

    let mut state = ctx.create_state().map_err(|e| format!("Erreur creation state: {:?}", e))?;
    state.full(params, &samples_f32).map_err(|e| format!("Erreur inference Whisper: {:?}", e))?;

    let num_segments = state.full_n_segments();
    let mut full_text = String::new();
    let mut segments = Vec::new();

    for i in 0..num_segments {
        if let Some(seg) = state.get_segment(i) {
            let text = seg.to_str_lossy().map_err(|e| format!("Erreur texte: {:?}", e))?;
            let start = seg.start_timestamp() as f64 / 100.0;
            let end = seg.end_timestamp() as f64 / 100.0;

            let trimmed = text.trim();
            if !trimmed.is_empty() {
                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(trimmed);
                segments.push(TranscriptionSegment {
                    text: trimmed.to_string(),
                    start,
                    end,
                    speaker_id: None,
                });
            }
        }
    }

    Ok(TranscriptionResponse {
        text: full_text,
        segments,
    })
}

/// Linear interpolation resampling.
pub fn resample(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;
        let s = if idx + 1 < samples.len() {
            samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
        } else {
            samples[idx.min(samples.len() - 1)] as f64
        };
        out.push(s.round() as i16);
    }
    out
}

fn downmix_to_mono(samples: &[i16], channels: usize) -> Vec<i16> {
    samples
        .chunks(channels)
        .map(|frame| {
            let sum: i32 = frame.iter().map(|&s| s as i32).sum();
            (sum / channels as i32) as i16
        })
        .collect()
}

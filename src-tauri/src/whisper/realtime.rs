use std::path::Path;
use tokio::sync::mpsc;

/// Transcription events emitted to the UI — same interface as the old Mistral WebSocket.
#[derive(Debug, Clone)]
pub enum TranscriptionEvent {
    TextDelta { text: String },
    Segment { text: String, start: f64, end: f64 },
    Done { text: String },
    Error { message: String },
}

/// Messages sent from the audio loop to the Whisper inference thread.
enum AudioMsg {
    Chunk(Vec<i16>),
    End,
}

/// Handle for sending audio to the local real-time transcription.
pub struct RealtimeHandle {
    tx: mpsc::UnboundedSender<AudioMsg>,
}

impl RealtimeHandle {
    /// Send a chunk of i16 PCM samples (at source sample rate).
    pub fn send_audio(&self, samples: Vec<i16>) {
        let _ = self.tx.send(AudioMsg::Chunk(samples));
    }

    /// Signal end of audio input.
    pub fn end_audio(&self) {
        let _ = self.tx.send(AudioMsg::End);
    }
}

/// Chunk duration in seconds for real-time processing.
const CHUNK_DURATION_SECS: f64 = 5.0;
/// Overlap in seconds between consecutive chunks to avoid word cutoff.
const OVERLAP_SECS: f64 = 1.0;
/// Sample rate expected by Whisper.
const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Start a local real-time transcription session using Whisper.
///
/// Audio is accumulated and processed in chunks of ~5 seconds.
/// Returns a `RealtimeHandle` for sending audio and a receiver for transcription events.
pub fn connect_realtime_local(
    model_path: &Path,
    source_sample_rate: u32,
) -> Result<
    (RealtimeHandle, mpsc::UnboundedReceiver<TranscriptionEvent>),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<AudioMsg>();
    let (event_tx, event_rx) = mpsc::unbounded_channel::<TranscriptionEvent>();

    let model_path = model_path.to_path_buf();
    let src_rate = source_sample_rate;

    // Spawn a dedicated OS thread for CPU-intensive Whisper inference
    std::thread::Builder::new()
        .name("whisper-realtime".into())
        .spawn(move || {
            realtime_worker(model_path, src_rate, audio_rx, event_tx);
        })?;

    Ok((RealtimeHandle { tx: audio_tx }, event_rx))
}

/// Worker that runs on a dedicated thread, accumulating audio and running Whisper inference.
fn realtime_worker(
    model_path: std::path::PathBuf,
    source_sample_rate: u32,
    audio_rx: mpsc::UnboundedReceiver<AudioMsg>,
    event_tx: mpsc::UnboundedSender<TranscriptionEvent>,
) {
    // Load model
    let ctx = match whisper_rs::WhisperContext::new_with_params(
        model_path.to_str().unwrap_or(""),
        whisper_rs::WhisperContextParameters::default(),
    ) {
        Ok(ctx) => ctx,
        Err(e) => {
            let _ = event_tx.send(TranscriptionEvent::Error {
                message: format!("Erreur chargement modele Whisper: {:?}", e),
            });
            return;
        }
    };

    // Create reusable state (avoid expensive re-init per chunk)
    let mut state = match ctx.create_state() {
        Ok(s) => s,
        Err(e) => {
            let _ = event_tx.send(TranscriptionEvent::Error {
                message: format!("Erreur creation state Whisper: {:?}", e),
            });
            return;
        }
    };

    eprintln!("[whisper-realtime] Modele et state charges avec succes");

    let chunk_samples = (CHUNK_DURATION_SECS * WHISPER_SAMPLE_RATE as f64) as usize;
    let overlap_samples = (OVERLAP_SECS * WHISPER_SAMPLE_RATE as f64) as usize;

    // Buffer of resampled 16kHz audio
    let mut buffer: Vec<i16> = Vec::new();
    // Total samples processed (for timestamp offset calculation)
    let mut total_processed: usize = 0;
    // Full accumulated text for the Done event
    let mut full_text = String::new();
    // Previous chunk's last text for deduplication
    let mut prev_tail: String = String::new();

    // We need a sync receiver — convert from tokio to blocking
    let mut audio_rx = audio_rx;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    loop {
        // Block until we get audio or end signal
        let msg = rt.block_on(audio_rx.recv());
        match msg {
            Some(AudioMsg::Chunk(samples)) => {
                // Resample to 16kHz if needed
                let resampled = if source_sample_rate != WHISPER_SAMPLE_RATE {
                    super::resample(&samples, source_sample_rate, WHISPER_SAMPLE_RATE)
                } else {
                    samples
                };
                buffer.extend_from_slice(&resampled);

                // Log buffer growth periodically
                if buffer.len() % 16000 == 0 || buffer.len() == resampled.len() {
                    let buffer_secs = buffer.len() as f64 / WHISPER_SAMPLE_RATE as f64;
                    if buffer.len() <= resampled.len() * 2 {
                        eprintln!(
                            "[whisper-realtime] Receiving audio: buffer={:.1}s, chunk_size={}, src_rate={}, need={}",
                            buffer_secs, resampled.len(), source_sample_rate, chunk_samples
                        );
                    }
                }

                // Process when we have enough audio
                while buffer.len() >= chunk_samples {
                    let chunk_end = chunk_samples;
                    let chunk: Vec<i16> = buffer[..chunk_end].to_vec();

                    // Calculate time offset for this chunk
                    let offset_secs = total_processed as f64 / WHISPER_SAMPLE_RATE as f64;

                    // Check audio level to avoid processing silence
                    let rms: f64 = (chunk.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / chunk.len() as f64).sqrt();

                    if rms < 50.0 {
                        eprintln!("[whisper-realtime] Chunk silencieux (rms={:.0}), skip inference", rms);
                        let advance = chunk_end - overlap_samples;
                        total_processed += advance;
                        buffer.drain(..advance);
                        continue;
                    }

                    eprintln!(
                        "[whisper-realtime] Inference chunk: offset={:.1}s, samples={}, rms={:.0}",
                        offset_secs, chunk.len(), rms
                    );
                    let t0 = std::time::Instant::now();

                    // Run inference
                    match super::transcribe_chunk_with_state(&mut state, &chunk, offset_secs) {
                        Ok(segments) => {
                            let elapsed = t0.elapsed();
                            eprintln!(
                                "[whisper-realtime] Inference terminee en {:.1}s, {} segments",
                                elapsed.as_secs_f64(), segments.len()
                            );
                            for seg in &segments {
                                // Simple dedup: skip if text matches the tail of previous chunk
                                let trimmed = seg.text.trim();
                                if !trimmed.is_empty() && trimmed != prev_tail.trim() {
                                    eprintln!("[whisper-realtime] Segment: '{}'", trimmed);
                                    // Emit as segment
                                    let _ = event_tx.send(TranscriptionEvent::Segment {
                                        text: trimmed.to_string(),
                                        start: seg.start,
                                        end: seg.end,
                                    });
                                    if !full_text.is_empty() {
                                        full_text.push(' ');
                                    }
                                    full_text.push_str(trimmed);
                                }
                            }
                            // Store last segment text for dedup
                            if let Some(last) = segments.last() {
                                prev_tail = last.text.clone();
                            }
                        }
                        Err(e) => {
                            eprintln!("[whisper-realtime] Erreur inference: {}", e);
                            let _ = event_tx.send(TranscriptionEvent::Error {
                                message: format!("Erreur transcription: {}", e),
                            });
                        }
                    }

                    // Advance buffer, keeping overlap for continuity
                    let advance = chunk_end - overlap_samples;
                    total_processed += advance;
                    buffer.drain(..advance);
                }
            }
            Some(AudioMsg::End) => {
                // Process remaining buffer
                if !buffer.is_empty() {
                    let offset_secs = total_processed as f64 / WHISPER_SAMPLE_RATE as f64;
                    match super::transcribe_chunk_with_state(&mut state, &buffer, offset_secs) {
                        Ok(segments) => {
                            for seg in &segments {
                                let trimmed = seg.text.trim();
                                if !trimmed.is_empty() && trimmed != prev_tail.trim() {
                                    let _ = event_tx.send(TranscriptionEvent::Segment {
                                        text: trimmed.to_string(),
                                        start: seg.start,
                                        end: seg.end,
                                    });
                                    if !full_text.is_empty() {
                                        full_text.push(' ');
                                    }
                                    full_text.push_str(trimmed);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[whisper-realtime] Erreur inference finale: {}", e);
                        }
                    }
                }

                let _ = event_tx.send(TranscriptionEvent::Done {
                    text: full_text,
                });
                break;
            }
            None => {
                // Channel closed
                let _ = event_tx.send(TranscriptionEvent::Done {
                    text: full_text,
                });
                break;
            }
        }
    }

    eprintln!("[whisper-realtime] Worker termine");
}

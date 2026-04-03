use std::path::Path;

/// A diarized segment with speaker label.
#[derive(Debug, Clone)]
pub struct DiarizedSegment {
    pub speaker: String,
    pub start: f64,
    pub end: f64,
}

/// Run speaker diarization on an audio file using sherpa-onnx.
///
/// Returns segments with speaker labels and timestamps.
pub fn diarize(
    audio_path: &Path,
    segmentation_model: &Path,
    embedding_model: &Path,
    num_speakers: Option<usize>,
) -> Result<Vec<DiarizedSegment>, Box<dyn std::error::Error + Send + Sync>> {
    // Read audio file and resample to 16kHz mono f32 if needed
    let samples = load_audio_16k_mono_f32(audio_path)?;

    let config = sherpa_rs::diarize::DiarizeConfig {
        num_clusters: num_speakers.map(|n| n as i32),
        ..Default::default()
    };

    let mut diarizer = sherpa_rs::diarize::Diarize::new(
        segmentation_model.to_str().ok_or("Chemin modele segmentation invalide")?,
        embedding_model.to_str().ok_or("Chemin modele embedding invalide")?,
        config,
    )
    .map_err(|e| format!("Erreur initialisation diarisation: {:?}", e))?;

    let segments = diarizer
        .compute(samples, None)
        .map_err(|e| format!("Erreur diarisation: {:?}", e))?;

    let result: Vec<DiarizedSegment> = segments
        .iter()
        .map(|seg| DiarizedSegment {
            speaker: format!("Speaker {}", seg.speaker + 1),
            start: seg.start as f64,
            end: seg.end as f64,
        })
        .collect();

    eprintln!(
        "[diarization] {} segments identifies, {} speakers",
        result.len(),
        result.iter().map(|s| s.speaker.as_str()).collect::<std::collections::HashSet<_>>().len()
    );

    Ok(result)
}

/// Merge transcription segments with diarization results.
///
/// For each transcription segment, find the diarized segment with the
/// greatest time overlap and assign its speaker label.
pub fn merge_with_transcription(
    transcription_segments: &[crate::whisper::TranscriptionSegment],
    diarized_segments: &[DiarizedSegment],
) -> Vec<crate::whisper::TranscriptionSegment> {
    transcription_segments
        .iter()
        .map(|tseg| {
            let speaker = find_best_speaker(tseg.start, tseg.end, diarized_segments);
            crate::whisper::TranscriptionSegment {
                text: tseg.text.clone(),
                start: tseg.start,
                end: tseg.end,
                speaker_id: speaker,
            }
        })
        .collect()
}

/// Load a WAV file, resample to 16kHz mono, and return f32 samples.
fn load_audio_16k_mono_f32(
    audio_path: &Path,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    // Read samples as i16
    let samples_i16: Vec<i16> = if spec.sample_format == hound::SampleFormat::Float {
        reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .map(|s| (s * 32767.0) as i16)
            .collect()
    } else {
        reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .collect()
    };

    // Downmix to mono if stereo
    let mono = if spec.channels > 1 {
        samples_i16
            .chunks(spec.channels as usize)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                (sum / spec.channels as i32) as i16
            })
            .collect()
    } else {
        samples_i16
    };

    // Resample to 16kHz if needed
    let resampled = if spec.sample_rate != 16000 {
        crate::whisper::resample(&mono, spec.sample_rate, 16000)
    } else {
        mono
    };

    // Convert to f32 normalized
    let f32_samples: Vec<f32> = resampled.iter().map(|&s| s as f32 / 32768.0).collect();

    eprintln!(
        "[diarization] Audio charge: {}Hz {}ch -> 16kHz mono, {} samples ({:.1}s)",
        spec.sample_rate,
        spec.channels,
        f32_samples.len(),
        f32_samples.len() as f64 / 16000.0
    );

    Ok(f32_samples)
}

/// Find the speaker with the most overlap for a given time range.
fn find_best_speaker(
    start: f64,
    end: f64,
    diarized: &[DiarizedSegment],
) -> Option<String> {
    let mut best_overlap = 0.0f64;
    let mut best_speaker: Option<String> = None;

    for seg in diarized {
        let overlap_start = start.max(seg.start);
        let overlap_end = end.min(seg.end);
        let overlap = (overlap_end - overlap_start).max(0.0);

        if overlap > best_overlap {
            best_overlap = overlap;
            best_speaker = Some(seg.speaker.clone());
        }
    }

    best_speaker
}

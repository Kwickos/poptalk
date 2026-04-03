use hound::{WavSpec, WavWriter, SampleFormat};
use std::path::Path;

/// Save PCM i16 samples to a WAV file (mono, specified sample rate)
pub fn save_wav(path: &Path, samples: &[i16], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_save_and_verify_wav() {
        let samples: Vec<i16> = (0..16000).map(|i| (i % 1000) as i16).collect();
        let path = env::temp_dir().join("poptalk_test_audio.wav");

        save_wav(&path, &samples, 16000).unwrap();

        assert!(path.exists());
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);

        // Verify we can read it back with hound
        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 16000);
        assert_eq!(spec.bits_per_sample, 16);

        let read_samples: Vec<i16> = reader.into_samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(read_samples.len(), 16000);
        assert_eq!(read_samples[0], samples[0]);
        assert_eq!(read_samples[999], samples[999]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_empty_wav() {
        let samples: Vec<i16> = vec![];
        let path = env::temp_dir().join("poptalk_test_empty.wav");

        save_wav(&path, &samples, 16000).unwrap();
        assert!(path.exists());

        let reader = hound::WavReader::open(&path).unwrap();
        let read_samples: Vec<i16> = reader.into_samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(read_samples.len(), 0);

        std::fs::remove_file(&path).ok();
    }
}

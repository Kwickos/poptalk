use crate::db::Segment;
use crate::llm::chat::Summary;

/// Formats a timestamp in seconds to `[MM:SS]` or `[HH:MM:SS]` if >= 1 hour.
fn format_timestamp(seconds: f64) -> String {
    let total_secs = seconds as u64;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    if hours > 0 {
        format!("[{:02}:{:02}:{:02}]", hours, minutes, secs)
    } else {
        format!("[{:02}:{:02}]", minutes, secs)
    }
}

/// Formats a duration in seconds to a human-readable string.
fn format_duration(duration_secs: f64) -> String {
    let total_secs = duration_secs as u64;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    if hours > 0 {
        format!("{}h {:02}min {:02}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}min {:02}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

/// Formats a session's transcript and summary into a Markdown document.
pub fn export_markdown(
    title: &str,
    date: &str,
    duration_secs: Option<f64>,
    segments: &[Segment],
    summary: &Option<Summary>,
) -> String {
    let mut md = String::new();

    // Header
    md.push_str(&format!("# {}\n\n", title));
    md.push_str(&format!("**Date:** {}\n", date));
    if let Some(dur) = duration_secs {
        md.push_str(&format!("**Duree:** {}\n", format_duration(dur)));
    }

    // Transcription section
    md.push_str("\n## Transcription\n\n");
    for segment in segments {
        let ts = format_timestamp(segment.start_time);
        if let Some(ref speaker) = segment.speaker {
            md.push_str(&format!("{} **{}:** {}\n", ts, speaker, segment.text));
        } else {
            md.push_str(&format!("{} {}\n", ts, segment.text));
        }
    }

    // Summary section (only if present)
    if let Some(ref summary) = summary {
        md.push_str("\n## Resume\n");

        if !summary.key_points.is_empty() {
            md.push_str("\n### Points cles\n");
            for point in &summary.key_points {
                md.push_str(&format!("- {}\n", point));
            }
        }

        if !summary.decisions.is_empty() {
            md.push_str("\n### Decisions\n");
            for decision in &summary.decisions {
                md.push_str(&format!("- {}\n", decision));
            }
        }

        if !summary.action_items.is_empty() {
            md.push_str("\n### Actions a suivre\n");
            for item in &summary.action_items {
                if let Some(ref assignee) = item.assignee {
                    md.push_str(&format!("- [ ] {} (Assignee: {})\n", item.description, assignee));
                } else {
                    md.push_str(&format!("- [ ] {}\n", item.description));
                }
            }
        }
    }

    md
}

/// Generates a PDF document from session data and saves it to the given path.
pub fn export_pdf(
    title: &str,
    date: &str,
    duration_secs: Option<f64>,
    segments: &[Segment],
    summary: &Option<Summary>,
    output_path: &std::path::Path,
) -> Result<(), String> {
    use genpdf::Element as _;

    let font_family = load_macos_fonts()?;

    let mut doc = genpdf::Document::new(font_family);
    doc.set_title(title);
    doc.set_minimal_conformance();

    let mut decorator = genpdf::SimplePageDecorator::new();
    decorator.set_margins(genpdf::Margins::all(25));
    doc.set_page_decorator(decorator);
    doc.set_font_size(10);

    // Title
    doc.push(genpdf::elements::Paragraph::new(title)
        .styled(genpdf::style::Style::new().bold().with_font_size(18)));
    doc.push(genpdf::elements::Break::new(1.5_f32));

    // Metadata
    doc.push(genpdf::elements::Paragraph::new(format!("Date : {}", date))
        .styled(genpdf::style::Style::new().with_font_size(10)
            .with_color(genpdf::style::Color::Rgb(100, 100, 100))));
    if let Some(dur) = duration_secs {
        doc.push(genpdf::elements::Paragraph::new(format!("Duree : {}", format_duration(dur)))
            .styled(genpdf::style::Style::new().with_font_size(10)
                .with_color(genpdf::style::Color::Rgb(100, 100, 100))));
    }
    doc.push(genpdf::elements::Break::new(2.0_f32));

    // Transcription header
    doc.push(genpdf::elements::Paragraph::new("Transcription")
        .styled(genpdf::style::Style::new().bold().with_font_size(14)));
    doc.push(genpdf::elements::Break::new(1.0_f32));

    // Segments
    for segment in segments {
        let ts = format_timestamp(segment.start_time);
        let mut para = genpdf::elements::Paragraph::default();
        para.push(genpdf::style::StyledString::new(
            format!("{} ", ts),
            genpdf::style::Style::new().with_font_size(9)
                .with_color(genpdf::style::Color::Rgb(120, 120, 120)),
        ));
        if let Some(ref speaker) = segment.speaker {
            para.push(genpdf::style::StyledString::new(
                format!("{} : ", speaker),
                genpdf::style::Style::new().bold().with_font_size(10),
            ));
        }
        para.push(genpdf::style::StyledString::new(
            segment.text.clone(),
            genpdf::style::Style::new().with_font_size(10),
        ));
        doc.push(para);
    }

    // Summary
    if let Some(ref summary) = summary {
        doc.push(genpdf::elements::Break::new(2.0_f32));
        doc.push(genpdf::elements::Paragraph::new("Resume")
            .styled(genpdf::style::Style::new().bold().with_font_size(14)));
        doc.push(genpdf::elements::Break::new(1.0_f32));

        if !summary.key_points.is_empty() {
            doc.push(genpdf::elements::Paragraph::new("Points cles")
                .styled(genpdf::style::Style::new().bold().with_font_size(12)));
            let mut list = genpdf::elements::UnorderedList::new();
            for point in &summary.key_points {
                list.push(genpdf::elements::Paragraph::new(point.as_str()));
            }
            doc.push(list);
        }

        if !summary.decisions.is_empty() {
            doc.push(genpdf::elements::Break::new(1.0_f32));
            doc.push(genpdf::elements::Paragraph::new("Decisions")
                .styled(genpdf::style::Style::new().bold().with_font_size(12)));
            let mut list = genpdf::elements::UnorderedList::new();
            for decision in &summary.decisions {
                list.push(genpdf::elements::Paragraph::new(decision.as_str()));
            }
            doc.push(list);
        }

        if !summary.action_items.is_empty() {
            doc.push(genpdf::elements::Break::new(1.0_f32));
            doc.push(genpdf::elements::Paragraph::new("Actions a suivre")
                .styled(genpdf::style::Style::new().bold().with_font_size(12)));
            let mut list = genpdf::elements::UnorderedList::new();
            for item in &summary.action_items {
                let text = if let Some(ref assignee) = item.assignee {
                    format!("{} (Assignee : {})", item.description, assignee)
                } else {
                    item.description.clone()
                };
                list.push(genpdf::elements::Paragraph::new(text));
            }
            doc.push(list);
        }
    }

    // Create parent directory
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Erreur creation dossier: {}", e))?;
    }

    doc.render_to_file(output_path)
        .map_err(|e| format!("Erreur generation PDF: {}", e))?;

    Ok(())
}

/// Load Arial font family from macOS system fonts.
fn load_macos_fonts() -> Result<genpdf::fonts::FontFamily<genpdf::fonts::FontData>, String> {
    let font_dir = std::path::Path::new("/System/Library/Fonts/Supplemental");

    let load = |filename: &str| -> Result<genpdf::fonts::FontData, String> {
        let path = font_dir.join(filename);
        let data = std::fs::read(&path)
            .map_err(|e| format!("Police '{}' introuvable: {}", path.display(), e))?;
        genpdf::fonts::FontData::new(data, None)
            .map_err(|e| format!("Erreur lecture police '{}': {}", filename, e))
    };

    let regular = load("Arial.ttf")?;
    let bold = load("Arial Bold.ttf")?;
    let italic = load("Arial Italic.ttf")?;
    let bold_italic = load("Arial Bold Italic.ttf")?;

    Ok(genpdf::fonts::FontFamily {
        regular,
        bold,
        italic,
        bold_italic,
    })
}

/// Writes content to a file at the given path.
pub fn export_to_file(content: &str, path: &std::path::Path) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Segment;
    use crate::mistral::chat::{ActionItem, Summary};

    fn make_segment(text: &str, start: f64, end: f64, speaker: Option<&str>) -> Segment {
        Segment {
            id: 1,
            session_id: "test-session".to_string(),
            text: text.to_string(),
            start_time: start,
            end_time: end,
            speaker: speaker.map(|s| s.to_string()),
            is_diarized: speaker.is_some(),
        }
    }

    #[test]
    fn test_export_markdown_with_speakers() {
        let segments = vec![
            make_segment("Bonjour a tous", 0.0, 5.0, Some("Speaker 1")),
            make_segment("Merci d'etre la", 5.0, 10.0, Some("Speaker 2")),
            make_segment("Commençons la reunion", 10.0, 15.0, Some("Speaker 1")),
        ];
        let summary = Some(Summary {
            key_points: vec![
                "Discussion du budget Q3".to_string(),
                "Revue du planning sprint".to_string(),
            ],
            decisions: vec!["Reporter la release d'une semaine".to_string()],
            action_items: vec![
                ActionItem {
                    description: "Mettre a jour le planning".to_string(),
                    assignee: Some("Alexandre".to_string()),
                },
                ActionItem {
                    description: "Envoyer le budget revise".to_string(),
                    assignee: None,
                },
            ],
        });

        let md = export_markdown(
            "Reunion Equipe",
            "2025-01-15",
            Some(900.0),
            &segments,
            &summary,
        );

        assert!(md.contains("# Reunion Equipe"));
        assert!(md.contains("**Date:** 2025-01-15"));
        assert!(md.contains("**Duree:** 15min 00s"));
        assert!(md.contains("[00:00] **Speaker 1:** Bonjour a tous"));
        assert!(md.contains("[00:05] **Speaker 2:** Merci d'etre la"));
        assert!(md.contains("[00:10] **Speaker 1:** Commençons la reunion"));
        assert!(md.contains("## Resume"));
        assert!(md.contains("### Points cles"));
        assert!(md.contains("- Discussion du budget Q3"));
        assert!(md.contains("- Revue du planning sprint"));
        assert!(md.contains("### Decisions"));
        assert!(md.contains("- Reporter la release d'une semaine"));
        assert!(md.contains("### Actions a suivre"));
        assert!(md.contains("- [ ] Mettre a jour le planning (Assignee: Alexandre)"));
        assert!(md.contains("- [ ] Envoyer le budget revise"));
    }

    #[test]
    fn test_export_markdown_without_summary() {
        let segments = vec![
            make_segment("Bonjour", 0.0, 2.0, Some("Speaker 1")),
            make_segment("Salut", 2.0, 4.0, Some("Speaker 2")),
        ];

        let md = export_markdown(
            "Reunion rapide",
            "2025-02-01",
            Some(60.0),
            &segments,
            &None,
        );

        assert!(md.contains("# Reunion rapide"));
        assert!(md.contains("## Transcription"));
        assert!(!md.contains("## Resume"));
        assert!(!md.contains("### Points cles"));
    }

    #[test]
    fn test_export_markdown_empty_segments() {
        let segments: Vec<Segment> = vec![];

        let md = export_markdown(
            "Reunion vide",
            "2025-03-01",
            None,
            &segments,
            &None,
        );

        assert!(md.contains("# Reunion vide"));
        assert!(md.contains("**Date:** 2025-03-01"));
        assert!(!md.contains("**Duree:**"));
        assert!(md.contains("## Transcription"));
        // After "## Transcription\n\n" there should be no segment lines
        let after_transcription = md.split("## Transcription\n\n").nth(1).unwrap_or("");
        assert!(after_transcription.is_empty());
    }

    #[test]
    fn test_export_markdown_timestamp_format() {
        let segments = vec![
            make_segment("Debut", 0.0, 5.0, None),
            make_segment("Milieu", 3599.0, 3605.0, None),
            make_segment("Apres une heure", 3661.0, 3700.0, Some("Speaker 1")),
            make_segment("Beaucoup plus tard", 7384.0, 7400.0, None),
        ];

        let md = export_markdown("Longue reunion", "2025-04-01", Some(7400.0), &segments, &None);

        // 0 seconds -> [00:00]
        assert!(md.contains("[00:00] Debut"));
        // 3599 seconds -> [59:59]
        assert!(md.contains("[59:59] Milieu"));
        // 3661 seconds -> [01:01:01]
        assert!(md.contains("[01:01:01] **Speaker 1:** Apres une heure"));
        // 7384 seconds -> [02:03:04]
        assert!(md.contains("[02:03:04] Beaucoup plus tard"));
        // Duration: 2h 03min 20s
        assert!(md.contains("**Duree:** 2h 03min 20s"));
    }

    #[test]
    fn test_export_to_file() {
        let dir = std::env::temp_dir().join("poptalk_test_export");
        let path = dir.join("test_export.md");
        let content = "# Test\n\nHello world";

        export_to_file(content, &path).unwrap();
        let read_back = std::fs::read_to_string(&path).unwrap();
        assert_eq!(read_back, content);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}

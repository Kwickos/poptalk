use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub key_points: Vec<String>,
    pub decisions: Vec<String>,
    pub action_items: Vec<ActionItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    pub description: String,
    pub assignee: Option<String>,
}

/// Default LLM API endpoint (Mistral, OpenAI-compatible).
const DEFAULT_BASE_URL: &str = "https://api.mistral.ai/v1";
const DEFAULT_MODEL: &str = "mistral-small-latest";

/// Build the full chat completions URL from a base URL.
fn chat_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    format!("{}/chat/completions", base)
}

/// Sends the transcript + user query to an LLM and returns a natural language answer.
pub async fn search_transcript(
    api_key: &str,
    transcript: &str,
    query: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();

    let messages = serde_json::json!([
        {
            "role": "system",
            "content": "Tu es un assistant qui repond a des questions sur une transcription de reunion. Reponds de maniere concise et precise en te basant uniquement sur la transcription fournie. Si l'information n'est pas dans la transcription, dis-le."
        },
        {
            "role": "user",
            "content": format!("Transcription de la reunion:\n\n{}\n\nQuestion: {}", transcript, query)
        }
    ]);

    let body = serde_json::json!({
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1000
    });

    let response = client
        .post(chat_url(DEFAULT_BASE_URL))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("LLM API error {}: {}", status, body).into());
    }

    let result: serde_json::Value = response.json().await?;
    let answer = result["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("Pas de reponse")
        .to_string();

    Ok(answer)
}

/// Sends the full diarized transcript to the LLM and returns a structured Summary.
pub async fn generate_summary(
    api_key: &str,
    transcript: &str,
) -> Result<Summary, Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();

    let messages = serde_json::json!([
        {
            "role": "system",
            "content": "Tu es un assistant specialise dans la synthese de reunions. A partir de la transcription fournie, genere un resume structure au format JSON avec les champs suivants:\n- key_points: liste des points cles discutes\n- decisions: liste des decisions prises\n- action_items: liste des actions a mener, chacune avec 'description' et 'assignee' (null si non identifie)\n\nReponds UNIQUEMENT avec le JSON, sans texte avant ou apres."
        },
        {
            "role": "user",
            "content": format!("Transcription de la reunion:\n\n{}", transcript)
        }
    ]);

    let body = serde_json::json!({
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"}
    });

    let response = client
        .post(chat_url(DEFAULT_BASE_URL))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("LLM API error {}: {}", status, body).into());
    }

    let result: serde_json::Value = response.json().await?;
    let content = result["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("{}");

    let summary: Summary = serde_json::from_str(content)?;
    Ok(summary)
}

/// Generates a short, descriptive title for a meeting based on the transcript.
pub async fn generate_title(
    api_key: &str,
    transcript: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();

    // Use only the first ~2000 chars to keep it fast and cheap
    let context: &str = if transcript.len() > 2000 {
        &transcript[..2000]
    } else {
        transcript
    };

    let messages = serde_json::json!([
        {
            "role": "system",
            "content": "Tu generes un titre court et descriptif pour une reunion a partir de sa transcription. Le titre doit faire maximum 6 mots, sans guillemets, sans ponctuation finale. Exemples: 'Point sprint backend API', 'Revue budget Q3 marketing', 'Onboarding nouveau designer'. Reponds UNIQUEMENT avec le titre, rien d'autre."
        },
        {
            "role": "user",
            "content": context
        }
    ]);

    let body = serde_json::json!({
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 30
    });

    let response = client
        .post(chat_url(DEFAULT_BASE_URL))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("LLM API error {}: {}", status, body).into());
    }

    let result: serde_json::Value = response.json().await?;
    let title = result["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("Reunion")
        .trim()
        .trim_matches('"')
        .to_string();

    Ok(title)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_summary() {
        let json = r#"{
            "key_points": ["Discussion du budget Q3", "Revue du planning sprint"],
            "decisions": ["Reporter la release d'une semaine"],
            "action_items": [
                {"description": "Mettre a jour le planning", "assignee": "Alexandre"},
                {"description": "Envoyer le budget revise", "assignee": null}
            ]
        }"#;

        let summary: Summary = serde_json::from_str(json).unwrap();
        assert_eq!(summary.key_points.len(), 2);
        assert_eq!(summary.decisions.len(), 1);
        assert_eq!(summary.action_items.len(), 2);
        assert_eq!(summary.action_items[0].assignee.as_deref(), Some("Alexandre"));
        assert!(summary.action_items[1].assignee.is_none());
    }

    #[test]
    fn test_deserialize_empty_summary() {
        let json = r#"{"key_points": [], "decisions": [], "action_items": []}"#;
        let summary: Summary = serde_json::from_str(json).unwrap();
        assert!(summary.key_points.is_empty());
    }

    #[test]
    fn test_chat_url() {
        assert_eq!(chat_url("https://api.mistral.ai/v1"), "https://api.mistral.ai/v1/chat/completions");
        assert_eq!(chat_url("https://api.mistral.ai/v1/"), "https://api.mistral.ai/v1/chat/completions");
        assert_eq!(chat_url("http://localhost:11434/v1"), "http://localhost:11434/v1/chat/completions");
    }
}

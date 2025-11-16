use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub mod features;
pub mod matcher;
pub mod model;

pub use matcher::Matcher;
pub use model::{binary_accuracy, binary_cross_entropy, DifficultyModel};

// ============================================================================
// Core Data Types
// ============================================================================

/// A single word match from Aho–Corasick
#[derive(Clone, Debug)]
pub struct WordMatch {
    pub text: String,
    pub hsk_level: Option<u32>,
    pub top_level: Option<u32>,
    pub start: usize,
    pub end: usize,
}

/// Feature vector for a sentence (12 dimensions)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SentenceFeatures {
    // HSK word-level stats
    pub hsk_max: f32,
    pub hsk_mean: f32,
    pub hsk_count_high: u32, // levels 5-6
    pub hsk_count_low: u32,  // levels 1-2

    // TOP word-level stats
    pub top_max: f32,
    pub top_mean: f32,
    pub top_count_high: u32, // levels 6-8

    // Text metrics
    pub total_words: u32,
    pub oov_ratio: f32,
    pub sentence_length: u32,
    pub rare_char_ratio: f32,
    pub avg_word_length: f32,
}

/// Single training example
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingExample {
    pub sentence: String, // CRITICAL: store actual sentence for debugging/validation
    pub features: SentenceFeatures,
    pub hsk_labels: Vec<u32>, // [L1, L2, L3, L4, L5, L6] ∈ {0, 1}
    pub top_labels: Vec<u32>, // [L1...L8] ∈ {0, 1}
}

/// Feature normalization statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureStats {
    pub hsk_max_mean: f32,
    pub hsk_max_std: f32,
    pub hsk_mean_mean: f32,
    pub hsk_mean_std: f32,
    pub hsk_count_high_mean: f32,
    pub hsk_count_high_std: f32,
    pub hsk_count_low_mean: f32,
    pub hsk_count_low_std: f32,
    pub top_max_mean: f32,
    pub top_max_std: f32,
    pub top_mean_mean: f32,
    pub top_mean_std: f32,
    pub top_count_high_mean: f32,
    pub top_count_high_std: f32,
    pub total_words_mean: f32,
    pub total_words_std: f32,
    pub oov_ratio_mean: f32,
    pub oov_ratio_std: f32,
    pub sentence_length_mean: f32,
    pub sentence_length_std: f32,
    pub rare_char_ratio_mean: f32,
    pub rare_char_ratio_std: f32,
    pub avg_word_length_mean: f32,
    pub avg_word_length_std: f32,
}

impl FeatureStats {
    /// Normalize a feature vector using z-score
    pub fn normalize(&self, feat: &SentenceFeatures) -> Vec<f32> {
        vec![
            self.normalize_f32(feat.hsk_max, self.hsk_max_mean, self.hsk_max_std),
            self.normalize_f32(feat.hsk_mean, self.hsk_mean_mean, self.hsk_mean_std),
            self.normalize_u32(
                feat.hsk_count_high,
                self.hsk_count_high_mean,
                self.hsk_count_high_std,
            ),
            self.normalize_u32(
                feat.hsk_count_low,
                self.hsk_count_low_mean,
                self.hsk_count_low_std,
            ),
            self.normalize_f32(feat.top_max, self.top_max_mean, self.top_max_std),
            self.normalize_f32(feat.top_mean, self.top_mean_mean, self.top_mean_std),
            self.normalize_u32(
                feat.top_count_high,
                self.top_count_high_mean,
                self.top_count_high_std,
            ),
            self.normalize_u32(
                feat.total_words,
                self.total_words_mean,
                self.total_words_std,
            ),
            self.normalize_f32(feat.oov_ratio, self.oov_ratio_mean, self.oov_ratio_std),
            self.normalize_u32(
                feat.sentence_length,
                self.sentence_length_mean,
                self.sentence_length_std,
            ),
            self.normalize_f32(
                feat.rare_char_ratio,
                self.rare_char_ratio_mean,
                self.rare_char_ratio_std,
            ),
            self.normalize_f32(
                feat.avg_word_length,
                self.avg_word_length_mean,
                self.avg_word_length_std,
            ),
        ]
    }

    fn normalize_f32(&self, val: f32, mean: f32, std: f32) -> f32 {
        if std < 1e-6 {
            0.0
        } else {
            (val - mean) / std
        }
    }

    fn normalize_u32(&self, val: u32, mean: f32, std: f32) -> f32 {
        self.normalize_f32(val as f32, mean, std)
    }
}

/// Training dataset
#[derive(Serialize, Deserialize)]
pub struct TrainingDataset {
    pub examples: Vec<TrainingExample>,
    pub feature_stats: FeatureStats,
}

/// Runtime scoring output
#[derive(Debug, Clone)]
pub struct DifficultyScore {
    pub sentence: String,
    pub hsk_level: u32,
    pub hsk_confidence: f32,
    pub hsk_probabilities: Vec<f32>,
    pub top_level: u32,
    pub top_confidence: f32,
    pub top_probabilities: Vec<f32>,
}

// ============================================================================
// Vocabulary & Matcher
// ============================================================================

/// Loads HSK vocabulary from JSON
/// Supports formats like: {entries: [{sm: "word", meanings: [{hsk: 1}]}]}
pub fn load_hsk_vocab(path: &Path) -> Result<FxHashMap<String, u32>> {
    let content = std::fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&content)?;

    let mut vocab = FxHashMap::default();

    // Try entries format first (used in most of our dictionaries)
    if let Some(entries) = data.get("entries").and_then(|e| e.as_array()) {
        for entry in entries {
            if let Some(word) = entry.get("sm").and_then(|w| w.as_str()) {
                if let Some(meanings) = entry.get("meanings").and_then(|m| m.as_array()) {
                    for meaning in meanings {
                        if let Some(hsk_level) = meaning.get("hsk").and_then(|l| l.as_u64()) {
                            vocab.insert(word.to_string(), hsk_level as u32);
                            break; // Take first HSK level found
                        }
                    }
                }
            }
        }
        return Ok(vocab);
    }

    // Fallback: handle {word: level} format
    match data {
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let (Some(word), Some(level)) = (
                    item.get("word").and_then(|w| w.as_str()),
                    item.get("level").and_then(|l| l.as_u64()),
                ) {
                    vocab.insert(word.to_string(), level as u32);
                }
            }
        }
        serde_json::Value::Object(obj) => {
            for (word, level_val) in obj {
                if let Some(level) = level_val.as_u64() {
                    vocab.insert(word, level as u32);
                }
            }
        }
        _ => return Err(anyhow!("Invalid HSK vocabulary format")),
    }

    Ok(vocab)
}

/// Loads TOP vocabulary from JSON
/// Supports formats like: {entries: [{sm: "word", meanings: [{top: 1}]}]}
pub fn load_top_vocab(path: &Path) -> Result<FxHashMap<String, u32>> {
    let content = std::fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&content)?;

    let mut vocab = FxHashMap::default();

    // Try entries format first
    if let Some(entries) = data.get("entries").and_then(|e| e.as_array()) {
        for entry in entries {
            if let Some(word) = entry.get("sm").and_then(|w| w.as_str()) {
                if let Some(meanings) = entry.get("meanings").and_then(|m| m.as_array()) {
                    for meaning in meanings {
                        if let Some(top_level) = meaning.get("top").and_then(|l| l.as_u64()) {
                            vocab.insert(word.to_string(), top_level as u32);
                            break; // Take first TOP level found
                        }
                    }
                }
            }
        }
        return Ok(vocab);
    }

    // Fallback: handle {word: level} format
    match data {
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let (Some(word), Some(level)) = (
                    item.get("word").and_then(|w| w.as_str()),
                    item.get("level").and_then(|l| l.as_u64()),
                ) {
                    vocab.insert(word.to_string(), level as u32);
                }
            }
        }
        serde_json::Value::Object(obj) => {
            for (word, level_val) in obj {
                if let Some(level) = level_val.as_u64() {
                    vocab.insert(word, level as u32);
                }
            }
        }
        _ => return Err(anyhow!("Invalid TOP vocabulary format")),
    }

    Ok(vocab)
}

// NOTE: Tokenization is now handled by Matcher::find_matches() using Aho–Corasick
// See matcher.rs for the correct implementation

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_stats_normalization() {
        let stats = FeatureStats {
            hsk_max_mean: 3.0,
            hsk_max_std: 1.0,
            hsk_mean_mean: 2.5,
            hsk_mean_std: 0.8,
            hsk_count_high_mean: 2.0,
            hsk_count_high_std: 1.0,
            hsk_count_low_mean: 2.0,
            hsk_count_low_std: 1.0,
            top_max_mean: 3.0,
            top_max_std: 1.0,
            top_mean_mean: 2.5,
            top_mean_std: 0.8,
            top_count_high_mean: 0.5,
            top_count_high_std: 0.5,
            total_words_mean: 8.0,
            total_words_std: 3.0,
            oov_ratio_mean: 0.1,
            oov_ratio_std: 0.05,
            sentence_length_mean: 20.0,
            sentence_length_std: 8.0,
            rare_char_ratio_mean: 0.05,
            rare_char_ratio_std: 0.03,
            avg_word_length_mean: 2.0,
            avg_word_length_std: 0.5,
        };

        let feat = SentenceFeatures {
            hsk_max: 3.0,
            hsk_mean: 2.5,
            hsk_count_high: 2,
            hsk_count_low: 2,
            top_max: 3.0,
            top_mean: 2.5,
            top_count_high: 0,
            total_words: 8,
            oov_ratio: 0.1,
            sentence_length: 20,
            rare_char_ratio: 0.05,
            avg_word_length: 2.0,
        };

        let normalized = stats.normalize(&feat);
        assert_eq!(normalized.len(), 12);
        assert!(normalized[0] == 0.0); // hsk_max normalized
    }
}

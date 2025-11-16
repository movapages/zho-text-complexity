use crate::{SentenceFeatures, WordMatch};
use anyhow::Result;

// ============================================================================
// Feature Extraction
// ============================================================================

/// Extract 12-dimensional feature vector from word matches
///
/// Features:
/// 1. hsk_max: maximum HSK level found
/// 2. hsk_mean: mean HSK level
/// 3. hsk_count_high: count of HSK levels 5-6
/// 4. hsk_count_low: count of HSK levels 1-2
/// 5. top_max: maximum TOP level found
/// 6. top_mean: mean TOP level
/// 7. top_count_high: count of TOP level 4 words (TOP dictionary only has levels 1-4)
/// 8. total_words: number of matched words
/// 9. oov_ratio: out-of-vocabulary ratio
/// 10. sentence_length: character count
/// 11. rare_char_ratio: estimated rare character proportion
/// 12. avg_word_length: average matched word length
pub fn extract_features(matches: &[WordMatch], sentence: &str) -> Result<SentenceFeatures> {
    if matches.is_empty() {
        return Ok(SentenceFeatures {
            hsk_max: 0.0,
            hsk_mean: 0.0,
            hsk_count_high: 0,
            hsk_count_low: 0,
            top_max: 0.0,
            top_mean: 0.0,
            top_count_high: 0,
            total_words: 0,
            oov_ratio: 1.0,
            sentence_length: sentence.chars().count() as u32, // FIX: count characters, not bytes
            rare_char_ratio: 0.0,
            avg_word_length: 0.0,
        });
    }

    // Coverage metrics
    let total_words = matches.len() as u32;
    let matched_chars: usize = matches.iter().map(|m| m.text.chars().count()).sum();
    let sentence_length = sentence.chars().count() as u32;
    let unmatched_chars = (sentence_length as usize).saturating_sub(matched_chars);

    // OOV ratio: percentage of characters NOT matched by dictionaries
    let oov_ratio = if sentence_length > 0 {
        unmatched_chars as f32 / sentence_length as f32
    } else {
        0.0
    };

    // HSK statistics: ONLY use matched words, do NOT inflate with OOV
    // OOV is captured separately in oov_ratio feature
    // Inflating with level 7 for every unmatched character (including punctuation)
    // causes catastrophic label distribution where even easy sentences appear HSK 6-7
    let hsk_levels: Vec<f32> = matches
        .iter()
        .filter_map(|m| m.hsk_level.map(|l| l as f32))
        .collect();

    let (hsk_max, hsk_mean, hsk_count_high, hsk_count_low) = compute_level_stats(&hsk_levels, 5, 2);

    // TOP statistics: Only use matched TOP words (do NOT inflate with OOV)
    // TOP dictionary only contains levels 1-4
    // OOV will be handled in label computation, not features
    let top_levels: Vec<f32> = matches
        .iter()
        .filter_map(|m| m.top_level.map(|l| l as f32))
        .collect();

    // FIX: Changed threshold from 6 to 4, since TOP only has levels 1-4
    // top_count_high now counts TOP-4 words (hardest words in the TOP dictionary)
    let (top_max, top_mean, top_count_high, _) = compute_level_stats(&top_levels, 4, 1);

    // Average word length (in characters)
    let avg_word_length = if total_words > 0 {
        matched_chars as f32 / total_words as f32
    } else {
        0.0
    };

    // Estimate rare character ratio
    // This is a heuristic: characters are "rare" if they only appear in level 5+ words
    let rare_char_ratio = estimate_rare_char_ratio(matches);

    Ok(SentenceFeatures {
        hsk_max,
        hsk_mean,
        hsk_count_high,
        hsk_count_low,
        top_max,
        top_mean,
        top_count_high,
        total_words,
        oov_ratio,
        sentence_length,
        rare_char_ratio,
        avg_word_length,
    })
}

/// Compute statistics for a level array
fn compute_level_stats(
    levels: &[f32],
    high_threshold: u32,
    low_threshold: u32,
) -> (f32, f32, u32, u32) {
    if levels.is_empty() {
        return (0.0, 0.0, 0, 0);
    }

    let max = levels.iter().cloned().fold(0.0, f32::max);
    let mean = levels.iter().sum::<f32>() / levels.len() as f32;

    let count_high = levels
        .iter()
        .filter(|&&l| l >= high_threshold as f32)
        .count() as u32;

    let count_low = levels
        .iter()
        .filter(|&&l| l <= low_threshold as f32)
        .count() as u32;

    (max, mean, count_high, count_low)
}

/// Estimate proportion of rare characters (heuristic)
fn estimate_rare_char_ratio(matches: &[WordMatch]) -> f32 {
    let high_level_words: Vec<&WordMatch> = matches
        .iter()
        .filter(|m| m.hsk_level.map_or(false, |l| l >= 5) || m.top_level.map_or(false, |l| l >= 4))
        .collect();

    if high_level_words.is_empty() {
        return 0.0;
    }

    let high_level_chars: usize = high_level_words
        .iter()
        .map(|m| m.text.chars().count())
        .sum(); // FIX: count characters
    let total_chars: usize = matches.iter().map(|m| m.text.chars().count()).sum(); // FIX: count characters

    if total_chars == 0 {
        0.0
    } else {
        high_level_chars as f32 / total_chars as f32
    }
}

// ============================================================================
// Coverage & Labeling (90% Rule)
// ============================================================================

/// Compute lexical coverage for a specific learner level
///
/// coverage = known_words / total_words
/// where known_words have difficulty ≤ level
pub fn compute_coverage(matches: &[WordMatch], level: u32, is_hsk: bool) -> f32 {
    if matches.is_empty() {
        return 0.0;
    }

    let known = matches
        .iter()
        .filter(|m| {
            if is_hsk {
                m.hsk_level.map_or(false, |l| l <= level)
            } else {
                m.top_level.map_or(false, |l| l <= level)
            }
        })
        .count();

    known as f32 / matches.len() as f32
}

/// Generate training labels using the 90% comprehension rule for HSK
/// and a stricter max-level-aware rule for TOP.
///
/// Returns (hsk_labels, top_labels)
/// - hsk_labels: [L1..L7] where 1=readable, 0=not readable (L7 = OOV / very hard)
/// - top_labels: [L1..L5] where 1=readable, 0=not readable (L5 = beyond TOP, TOP has only 1-4)
pub fn compute_labels(matches: &[WordMatch]) -> Result<(Vec<u32>, Vec<u32>)> {
    let hsk_labels = compute_hsk_labels(matches);
    let top_labels = compute_top_labels(matches);
    Ok((hsk_labels, top_labels))
}

/// HSK: keep simple coverage-based rule with monotonic closure.
fn compute_hsk_labels(matches: &[WordMatch]) -> Vec<u32> {
    let coverage_threshold = 0.90;

    // HSK: 7 levels (including L7 for OOV words / very hard)
    let mut hsk_labels: Vec<u32> = (1..=7)
        .map(|level| {
            let coverage = compute_coverage(matches, level, true);
            if coverage >= coverage_threshold {
                1
            } else {
                0
            }
        })
        .collect();

    enforce_monotonic(&mut hsk_labels);
    hsk_labels
}

/// TOP: use ordinal labels based on max TOP level found
/// Note: TOP dictionary only contains levels 1-4, so we have 5 total levels (1-4 + 5 for OOV)
///
/// Ordinal labeling (CORAL):
/// - If max_top = 1 → readable from L1 onwards: [1,1,1,1,1]
/// - If max_top = 3 → readable from L3 onwards: [0,0,1,1,1]  
/// - If max_top = 5 (OOV) → readable from L5 only: [0,0,0,0,1]
fn compute_top_labels(matches: &[WordMatch]) -> Vec<u32> {
    let mut labels = vec![0u32; 5]; // TOP has only 4 levels + 1 for beyond-TOP

    // Count TOP-matched words
    let _total_matches = matches.len();
    let top_matched: Vec<u32> = matches.iter().filter_map(|m| m.top_level).collect();

    // Determine max TOP level
    let max_top = if top_matched.is_empty() {
        // No TOP matches at all → beyond TOP (level 5)
        5
    } else {
        // Use max TOP level from matched words, in range 1–4
        top_matched.iter().copied().max().unwrap_or(4).min(4)
    };

    // CORAL ordinal labeling: label[k] = 1 if max_top <= k
    // This creates cumulative probabilities naturally
    for level in 1..=5 {
        if max_top <= level {
            labels[(level - 1) as usize] = 1;
        }
    }

    labels
}

/// Enforce monotonic constraint on labels
/// If label[i] == 1, then label[i+1..] must all be 1
fn enforce_monotonic(labels: &mut [u32]) {
    for i in 1..labels.len() {
        if labels[i - 1] == 1 {
            labels[i] = 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_match(text: &str, hsk: Option<u32>, top: Option<u32>) -> WordMatch {
        WordMatch {
            text: text.to_string(),
            hsk_level: hsk,
            top_level: top,
            start: 0,
            end: text.len(),
        }
    }

    #[test]
    fn test_compute_coverage() {
        let matches = vec![
            make_match("我", Some(1), Some(1)),
            make_match("喜欢", Some(3), Some(1)),
            make_match("吃", Some(1), Some(1)),
            make_match("甜", Some(3), Some(2)),
            make_match("苹果", Some(2), Some(1)),
            make_match("和", Some(2), Some(1)),
            make_match("新鲜", Some(3), Some(2)),
            make_match("葡萄", Some(3), Some(1)),
        ];

        // HSK Level 1: only "我", "吃" = 2/8 = 25%
        assert!((compute_coverage(&matches, 1, true) - 0.25).abs() < 0.01);

        // HSK Level 3: all 8 words = 8/8 = 100%
        assert!((compute_coverage(&matches, 3, true) - 1.0).abs() < 0.01);

        // TOP Level 1: 6 words = 6/8 = 75%
        assert!((compute_coverage(&matches, 1, false) - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_compute_labels() {
        let matches = vec![
            make_match("我", Some(1), Some(1)),
            make_match("喜欢", Some(3), Some(1)),
            make_match("吃", Some(1), Some(1)),
            make_match("苹果", Some(2), Some(1)),
        ];

        let (hsk_labels, top_labels) = compute_labels(&matches).unwrap();

        // HSK: L1 25%, L2 75%, L3+ 100% (now 7 levels including L7 for OOV)
        assert_eq!(hsk_labels, vec![0, 0, 1, 1, 1, 1, 1]);

        // TOP: L1 100% (all TOP level 1, now 5 levels: 1-4 + 5 for beyond-TOP)
        assert_eq!(top_labels, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_extract_features() {
        let matches = vec![
            make_match("我", Some(1), Some(1)),
            make_match("喜欢", Some(3), Some(1)),
            make_match("吃", Some(1), Some(1)),
            make_match("甜", Some(3), Some(2)),
        ];

        let feat = extract_features(&matches, "我喜欢吃甜。").unwrap();

        assert_eq!(feat.hsk_max, 3.0);
        assert!((feat.hsk_mean - 2.0).abs() < 0.01);
        assert_eq!(feat.hsk_count_high, 0); // no levels 5-6
        assert_eq!(feat.hsk_count_low, 2); // "我", "吃"
        assert_eq!(feat.total_words, 4);
        assert_eq!(feat.oov_ratio, 0.0);
    }
}

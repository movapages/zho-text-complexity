use anyhow::Result;
use clap::Parser;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use zho_complexity::features::{compute_labels, extract_features};
use zho_complexity::{load_hsk_vocab, load_top_vocab, Matcher, TrainingDataset, TrainingExample};

#[derive(Parser, Debug)]
#[command(name = "prepare_training_data")]
#[command(about = "Generate training dataset from sentences")]
struct Args {
    /// Input sentences JSON file
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output training data JSON file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to HSK dictionary
    #[arg(long, default_value = "dictionaries/hsk_dictionary.json")]
    hsk_dict: PathBuf,

    /// Path to TOP dictionary
    #[arg(long, default_value = "dictionaries/top_dictionary.json")]
    top_dict: PathBuf,

    /// Number of example sentences to generate (if no input file)
    #[arg(long, default_value = "10")]
    num_examples: usize,

    /// Path to BKRS difficult phrases (optional - adds TOP 7-8 level content)
    #[arg(long, default_value = "data/bkrs_difficult_phrases.json")]
    bkrs_phrases: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input_path = args
        .input
        .unwrap_or_else(|| PathBuf::from("data/sentences.json"));
    let output_path = args
        .output
        .unwrap_or_else(|| PathBuf::from("data/training_data.json"));

    // Create data directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("[1/3] Loading vocabularies and building matcher...");
    let hsk_vocab = load_hsk_vocab(&args.hsk_dict)?;
    let top_vocab = load_top_vocab(&args.top_dict)?;
    let matcher = Matcher::new(hsk_vocab.clone(), top_vocab.clone())?;
    println!("  ✓ HSK: {} words", hsk_vocab.len());
    println!("  ✓ TOP: {} words", top_vocab.len());

    println!("\n[2/3] Loading sentences...");
    let mut sentences = if input_path.exists() {
        load_sentences_from_file(&input_path)?
    } else {
        println!("  ⚠ Input file not found, generating BALANCED synthetic dataset");
        Vec::new()
    };

    // Generate BALANCED training data with PURE level sentences
    // Each sentence targets EXACTLY one TOP level for clean ordinal labels
    // NOTE: TOP dictionary only has levels 1-4, BKRS phrases serve as "beyond TOP" (L5)

    println!("  → Generating TOP L1 sentences...");
    let l1_sentences = generate_level_targeted_sentences(600, &top_vocab, &hsk_vocab, 1, 1);
    println!("     Generated {} L1 sentences", l1_sentences.len());
    sentences.extend(l1_sentences);

    println!("  → Generating TOP L2 sentences...");
    let l2_sentences = generate_level_targeted_sentences(600, &top_vocab, &hsk_vocab, 2, 2);
    println!("     Generated {} L2 sentences", l2_sentences.len());
    sentences.extend(l2_sentences);

    println!("  → Generating TOP L3 sentences...");
    let l3_sentences = generate_level_targeted_sentences(400, &top_vocab, &hsk_vocab, 3, 3);
    println!("     Generated {} L3 sentences", l3_sentences.len());
    sentences.extend(l3_sentences);

    println!("  → Generating TOP L4 sentences...");
    let l4_sentences = generate_level_targeted_sentences(400, &top_vocab, &hsk_vocab, 4, 4);
    println!("     Generated {} L4 sentences", l4_sentences.len());
    sentences.extend(l4_sentences);

    // Add BKRS difficult phrases as "beyond TOP" (Level 5)
    if let Some(bkrs_path) = &args.bkrs_phrases {
        if bkrs_path.exists() {
            let mut bkrs_phrases = load_bkrs_phrases(bkrs_path)?;
            println!("  ✓ Loaded {} BKRS difficult phrases", bkrs_phrases.len());

            // Limit BKRS to 400 to balance with L1/L2/L3/L4 levels
            // This prevents hard examples from dominating the dataset
            bkrs_phrases.truncate(400);
            println!(
                "     Using {} BKRS phrases for TOP L5 (beyond TOP 1-4)",
                bkrs_phrases.len()
            );
            sentences.extend(bkrs_phrases);
        }
    }

    println!(
        "  ✓ Total {} sentences (balanced across difficulty levels)",
        sentences.len()
    );

    println!("\n[3/3] Extracting features & labels...");
    let mut examples = Vec::new();
    let mut feature_sums = [0.0; 12];
    let mut feature_sq_sums = [0.0; 12];

    for (idx, sentence) in sentences.iter().enumerate() {
        if (idx + 1) % 100 == 0 {
            println!("  Processing: {}/{}", idx + 1, sentences.len());
        }

        let matches = matcher.find_non_overlapping(sentence);

        // VALIDATION: Require at least 2 words (relaxed from 3)
        if matches.len() < 2 {
            continue;
        }

        if let Ok(features) = extract_features(&matches, sentence) {
            let (hsk_labels, top_labels) = compute_labels(&matches)?;

            // VALIDATION: Check label consistency - need at least ONE label
            let has_hsk_label = hsk_labels.iter().any(|&l| l == 1);
            let has_top_label = top_labels.iter().any(|&l| l == 1);

            if !has_hsk_label && !has_top_label {
                // Skip ONLY if BOTH are empty (relaxed)
                continue;
            }

            // VALIDATION: Sentence must be reasonable length (relaxed)
            if sentence.chars().count() < 5 || sentence.chars().count() > 200 {
                continue;
            }

            // VALIDATION: Word coverage - require very good dictionary coverage
            // Synthetic sentences should have minimal OOV (mostly punctuation only)
            if features.oov_ratio > 0.15 {
                continue;
            }

            // VALIDATION: Ensure features match vocabulary levels reasonably
            // This catches any sentences that slipped through with high-level words
            // HSK max should be reasonable (allow some flexibility for particle mixing)
            if features.hsk_max > 5.0 && features.top_max <= 2.0 {
                // TOP L1-L2 sentences shouldn't have HSK 6+ words
                continue;
            }

            if features.hsk_max > 6.0 && features.top_max <= 4.0 {
                // TOP L3-L4 sentences shouldn't have HSK 7 words
                continue;
            }

            // Accumulate for stats
            let feat_vec = vec![
                features.hsk_max,
                features.hsk_mean,
                features.hsk_count_high as f32,
                features.hsk_count_low as f32,
                features.top_max,
                features.top_mean,
                features.top_count_high as f32,
                features.total_words as f32,
                features.oov_ratio,
                features.sentence_length as f32,
                features.rare_char_ratio,
                features.avg_word_length,
            ];

            for (i, &val) in feat_vec.iter().enumerate() {
                feature_sums[i] += val;
                feature_sq_sums[i] += val * val;
            }

            examples.push(TrainingExample {
                sentence: sentence.clone(),
                features,
                hsk_labels,
                top_labels,
            });
        }
    }

    println!("  ✓ Extracted {} examples", examples.len());

    // Compute statistics
    let n = examples.len() as f32;
    let feature_stats = zho_complexity::FeatureStats {
        hsk_max_mean: feature_sums[0] / n,
        hsk_max_std: ((feature_sq_sums[0] / n) - (feature_sums[0] / n).powi(2)).sqrt(),
        hsk_mean_mean: feature_sums[1] / n,
        hsk_mean_std: ((feature_sq_sums[1] / n) - (feature_sums[1] / n).powi(2)).sqrt(),
        hsk_count_high_mean: feature_sums[2] / n,
        hsk_count_high_std: ((feature_sq_sums[2] / n) - (feature_sums[2] / n).powi(2)).sqrt(),
        hsk_count_low_mean: feature_sums[3] / n,
        hsk_count_low_std: ((feature_sq_sums[3] / n) - (feature_sums[3] / n).powi(2)).sqrt(),
        top_max_mean: feature_sums[4] / n,
        top_max_std: ((feature_sq_sums[4] / n) - (feature_sums[4] / n).powi(2)).sqrt(),
        top_mean_mean: feature_sums[5] / n,
        top_mean_std: ((feature_sq_sums[5] / n) - (feature_sums[5] / n).powi(2)).sqrt(),
        top_count_high_mean: feature_sums[6] / n,
        top_count_high_std: ((feature_sq_sums[6] / n) - (feature_sums[6] / n).powi(2)).sqrt(),
        total_words_mean: feature_sums[7] / n,
        total_words_std: ((feature_sq_sums[7] / n) - (feature_sums[7] / n).powi(2)).sqrt(),
        oov_ratio_mean: feature_sums[8] / n,
        oov_ratio_std: ((feature_sq_sums[8] / n) - (feature_sums[8] / n).powi(2)).sqrt(),
        sentence_length_mean: feature_sums[9] / n,
        sentence_length_std: ((feature_sq_sums[9] / n) - (feature_sums[9] / n).powi(2)).sqrt(),
        rare_char_ratio_mean: feature_sums[10] / n,
        rare_char_ratio_std: ((feature_sq_sums[10] / n) - (feature_sums[10] / n).powi(2)).sqrt(),
        avg_word_length_mean: feature_sums[11] / n,
        avg_word_length_std: ((feature_sq_sums[11] / n) - (feature_sums[11] / n).powi(2)).sqrt(),
    };

    let dataset = TrainingDataset {
        examples,
        feature_stats,
    };

    // Save to file
    let json = serde_json::to_string_pretty(&dataset)?;
    fs::write(&output_path, json)?;

    println!(
        "\n✓ Saved {} training examples to {:?}",
        dataset.examples.len(),
        output_path
    );
    print_stats(&dataset);

    Ok(())
}

fn load_sentences_from_file(path: &std::path::Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&content)?;

    let mut sentences = Vec::new();

    // Handle array of objects with "text" field
    if let Some(arr) = data.as_array() {
        for item in arr {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                sentences.push(text.to_string());
            }
        }
    }

    Ok(sentences)
}

#[derive(Deserialize)]
struct BkrsPhrase {
    text: String,
    #[allow(dead_code)]
    text_traditional: Option<String>,
    #[allow(dead_code)]
    explanation: String,
    #[allow(dead_code)]
    category: String,
}

fn load_bkrs_phrases(path: &std::path::Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let phrases: Vec<BkrsPhrase> = serde_json::from_str(&content)?;
    // Just extract the text field - both simplified and traditional are already separate entries
    Ok(phrases.into_iter().map(|p| p.text).collect())
}

/// Generate sentences targeting specific TOP difficulty levels
/// This ensures balanced label distribution across all levels
fn generate_level_targeted_sentences(
    count: usize,
    top_vocab: &rustc_hash::FxHashMap<String, u32>,
    hsk_vocab: &rustc_hash::FxHashMap<String, u32>,
    min_level: u32,
    max_level: u32,
) -> Vec<String> {
    use rand::seq::SliceRandom;
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut sentences = Vec::new();

    // Collect words in target level range, filtered by BOTH TOP and HSK
    // CRITICAL: Enforce STRICT purity - TOP L1 must be HSK 1, TOP L2 must be HSK 1-2, etc.
    // This prevents contradictory labels where TOP says "easy" but HSK says "hard"
    let mut words_by_level: Vec<Vec<String>> = vec![Vec::new(); 10]; // 0-9
    for (word, top_level) in top_vocab {
        if *top_level >= min_level && *top_level <= max_level {
            // STRICT: Only include word if HSK level <= TOP max_level
            if let Some(hsk_level) = hsk_vocab.get(word) {
                // TOP L1 → only HSK 1
                // TOP L2 → only HSK 1-2
                // TOP L3 → only HSK 1-3
                // TOP L4 → only HSK 1-4
                if *hsk_level <= max_level {
                    words_by_level[*top_level as usize].push(word.clone());
                }
            }
        }
    }

    // Common particles and function words (very frequent)
    let particles = vec![
        "的", "了", "在", "是", "很", "也", "都", "和", "有", "不", "吗", "呢", "吧", "着", "我",
        "你", "他", "她", "们", "这", "那", "什么", "怎么",
    ];

    // TOP dictionary only has levels 1-4, so we only generate for those
    // BKRS phrases will serve as "beyond TOP" examples
    if max_level > 4 {
        eprintln!(
            "     ⚠ Warning: TOP dictionary only has levels 1-4, cannot generate L{}+",
            max_level
        );
        return sentences;
    }

    // FOR ALL TOP LEVELS (1-4): Use STRICT level filtering
    // CRITICAL: Only use words from EXACTLY the target level range, NO leakage from higher levels
    let mut attempts = 0;
    let max_attempts = count * 50;

    // Build STRICT eligible word pools - ONLY words in [min_level, max_level]
    let mut eligible_pools: Vec<Vec<String>> = Vec::new();
    for lvl in min_level..=max_level {
        if lvl < 10 && !words_by_level[lvl as usize].is_empty() {
            eligible_pools.push(words_by_level[lvl as usize].clone());
        }
    }

    // Flatten into single pool for easy random selection
    let mut word_pool: Vec<String> = Vec::new();
    for pool in &eligible_pools {
        word_pool.extend(pool.iter().cloned());
    }

    if word_pool.is_empty() {
        eprintln!(
            "     ⚠ Warning: No words available for levels {}-{}",
            min_level, max_level
        );
        return sentences;
    }

    while sentences.len() < count && attempts < max_attempts {
        attempts += 1;

        // Progress indicator every 100 attempts
        if attempts % 100 == 0 {
            eprint!(
                "\r     Progress: {}/{} sentences (attempts: {})",
                sentences.len(),
                count,
                attempts
            );
        }

        // Sentence length: shorter for easier levels, longer for harder
        let length = if max_level <= 2 {
            rng.gen_range(3..7)
        } else if max_level <= 4 {
            rng.gen_range(5..10)
        } else {
            rng.gen_range(6..12)
        };

        let mut words = Vec::new();

        for _ in 0..length {
            // Allow particles for easy levels (TOP <= 4)
            if max_level <= 4 {
                if rng.gen_bool(0.3) {
                    if let Some(particle) = particles.choose(&mut rng) {
                        words.push(particle.to_string());
                        continue;
                    }
                }
            }

            // STRICT: Pick word ONLY from the allowed level range
            if let Some(word) = word_pool.choose(&mut rng) {
                words.push(word.clone());
            }
        }

        // For easy/intermediate levels: simple validation
        if words.len() >= 3 {
            let ending = if rng.gen_bool(0.15) {
                "？"
            } else if rng.gen_bool(0.1) {
                "！"
            } else {
                "。"
            };
            let sentence = words.join("") + ending;

            // Reasonable length bounds
            let char_count = sentence.chars().count();
            if char_count >= 5 && char_count <= 80 {
                sentences.push(sentence);
            }
        }
    }

    eprintln!(); // Clear progress line

    if sentences.len() < count {
        eprintln!(
            "     ⚠ Warning: Only generated {} out of {} requested sentences after {} attempts",
            sentences.len(),
            count,
            attempts
        );
    }

    sentences
}

// fn generate_example_sentences(
//     count: usize,
//     hsk_vocab: &rustc_hash::FxHashMap<String, u32>,
//     _top_vocab: &rustc_hash::FxHashMap<String, u32>,
// ) -> Vec<String> {
//     use rand::seq::SliceRandom;
//     use rand::Rng;
//     use regex::Regex;

//     let mut rng = rand::thread_rng();
//     let mut sentences = Vec::new();

//     // Try to load Taiwan dictionary for real example sentences
//     if let Ok(content) = std::fs::read_to_string("dictionaries/taiwan_with_simplified.json") {
//         if let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
//             // Regex to extract sentences in 「...」 quotes
//             let re = Regex::new(r"「([^」]+)」").unwrap();

//             for entry in &entries {
//                 if sentences.len() >= count * 10 {
//                     // Extract 10x to allow heavy filtering
//                     break;
//                 }

//                 if let Some(meaning) = entry.get("meaning").and_then(|m| m.as_str()) {
//                     // Extract all examples from meaning field
//                     for cap in re.captures_iter(meaning) {
//                         if sentences.len() >= count * 10 {
//                             break;
//                         }

//                         let sentence = cap.get(1).map(|m| m.as_str()).unwrap_or("");

//                         // RELAXED FILTERING to extract MORE examples:
//                         // 1. 10-150 chars (wider range)
//                         // 2. At least 50% Chinese (not 70%)
//                         // 3. At least 4 Chinese chars (not 5)

//                         if sentence.len() < 10 || sentence.len() > 150 {
//                             continue;
//                         }

//                         let chinese_count = sentence
//                             .chars()
//                             .filter(|c| {
//                                 let code = *c as u32;
//                                 code >= 0x4E00 && code <= 0x9FFF
//                             })
//                             .count();

//                         let total_chars = sentence.chars().count();

//                         // At least 50% Chinese characters (relaxed from 70%)
//                         if total_chars == 0 || (chinese_count as f32 / total_chars as f32) < 0.5 {
//                             continue;
//                         }

//                         // Must have at least 4 Chinese characters (relaxed from 5)
//                         if chinese_count < 4 {
//                             continue;
//                         }

//                         sentences.push(sentence.to_string());
//                     }
//                 }
//             }
//         }
//     }

//     // If we couldn't extract enough real examples, fill with synthetics
//     if sentences.len() < count {
//         println!(
//             "  ⚠ Extracted {} real examples from Taiwan dictionary, generating {} more",
//             sentences.len(),
//             count - sentences.len()
//         );

//         let mut hsk_by_level: Vec<Vec<String>> = vec![Vec::new(); 7];
//         for (word, level) in hsk_vocab {
//             if *level <= 6 {
//                 hsk_by_level[*level as usize].push(word.clone());
//             }
//         }

//         let particles = vec!["的", "了", "在", "是", "很", "也", "都"];

//         while sentences.len() < count {
//             let length = rng.gen_range(4..12);
//             let mut words = Vec::new();
//             let target_level = rng.gen_range(1..5);

//             for _ in 0..length {
//                 if rng.gen_bool(0.1) && !particles.is_empty() {
//                     if let Some(word) = particles.choose(&mut rng) {
//                         words.push(word.to_string());
//                         continue;
//                     }
//                 }

//                 if !hsk_by_level[target_level].is_empty() {
//                     if let Some(word) = hsk_by_level[target_level].choose(&mut rng) {
//                         words.push(word.clone());
//                     }
//                 }
//             }

//             if !words.is_empty() {
//                 let ending = if rng.gen_bool(0.1) { "？" } else { "。" };
//                 let sentence = words.join("") + ending;
//                 if sentence.len() >= 4 && sentence.len() <= 200 {
//                     sentences.push(sentence);
//                 }
//             }
//         }
//     }

//     sentences.truncate(count);
//     sentences
// }

fn print_stats(dataset: &TrainingDataset) {
    println!("\n--- Feature Statistics ---");
    println!(
        "HSK Max:  mean={:.2}, std={:.2}",
        dataset.feature_stats.hsk_max_mean, dataset.feature_stats.hsk_max_std
    );
    println!(
        "HSK Mean: mean={:.2}, std={:.2}",
        dataset.feature_stats.hsk_mean_mean, dataset.feature_stats.hsk_mean_std
    );
    println!(
        "Total Words: mean={:.2}, std={:.2}",
        dataset.feature_stats.total_words_mean, dataset.feature_stats.total_words_std
    );
    println!(
        "OOV Ratio: mean={:.2}, std={:.2}",
        dataset.feature_stats.oov_ratio_mean, dataset.feature_stats.oov_ratio_std
    );

    // Label distribution
    println!("\n--- Label Distribution ---");
    let mut hsk_counts = [0; 7];
    let mut top_counts = [0; 5]; // TOP has 5 levels (1-4 + beyond)

    for example in &dataset.examples {
        for (i, &label) in example.hsk_labels.iter().enumerate() {
            if label == 1 {
                hsk_counts[i] += 1;
            }
        }
        for (i, &label) in example.top_labels.iter().enumerate() {
            if label == 1 {
                top_counts[i] += 1;
            }
        }
    }

    print!("HSK readable at level: ");
    for (i, &count) in hsk_counts.iter().enumerate() {
        let pct = count as f32 / dataset.examples.len() as f32 * 100.0;
        print!("[L{}:{:.0}%] ", i + 1, pct);
    }
    println!();

    print!("TOP readable at level: ");
    for (i, &count) in top_counts.iter().enumerate() {
        let pct = count as f32 / dataset.examples.len() as f32 * 100.0;
        print!("[L{}:{:.0}%] ", i + 1, pct);
    }
    println!();
}

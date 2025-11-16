use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use std::path::PathBuf;
use zho_complexity::{
    features::{compute_labels, extract_features},
    load_hsk_vocab, load_top_vocab, DifficultyModel, Matcher, TrainingDataset,
};

#[derive(Parser, Debug)]
#[command(name = "zho-complexity")]
#[command(about = "Score Chinese text difficulty using HSK/TOP vocabularies")]
struct Args {
    /// Chinese text to analyze
    #[arg(value_name = "SENTENCE")]
    sentence: String,

    /// HSK dictionary path
    #[arg(long, default_value = "dictionaries/hsk_dictionary.json")]
    hsk_dict: PathBuf,

    /// TOP dictionary path
    #[arg(long, default_value = "dictionaries/top_dictionary.json")]
    top_dict: PathBuf,

    /// Path to trained model (optional - if not provided, uses rule-based scoring)
    #[arg(long, default_value = "data/model.safetensors")]
    model: PathBuf,

    /// Path to training data (for feature normalization stats)
    #[arg(long, default_value = "data/robust_training_data.json")]
    training_data: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load vocabularies
    let hsk_vocab = load_hsk_vocab(&args.hsk_dict)?;
    let top_vocab = load_top_vocab(&args.top_dict)?;

    // Create matcher
    let matcher = Matcher::new(hsk_vocab, top_vocab)?;

    // Find matches in sentence
    let matches = matcher.find_non_overlapping(&args.sentence);

    if matches.is_empty() {
        println!("No vocabulary matches found in: {}", args.sentence);
        return Ok(());
    }

    // Extract features
    let features = extract_features(&matches, &args.sentence)?;

    // Display basic info
    println!("\n📖 Sentence: {}", args.sentence);
    println!("\n📊 Matches: {} words found", matches.len());

    println!("\n📈 Features:");
    println!("  HSK max: {:.1}", features.hsk_max);
    println!("  HSK mean: {:.2}", features.hsk_mean);
    println!("  TOP max: {:.1}", features.top_max);
    println!("  TOP mean: {:.2}", features.top_mean);
    println!("  Sentence length: {} chars", features.sentence_length);
    println!("  OOV ratio: {:.1}%", features.oov_ratio * 100.0);

    // Try to load and use trained model
    if args.model.exists() && args.training_data.exists() {
        println!("\n🧠 Neural Model Predictions:");

        // Load training data for normalization stats
        let training_json = std::fs::read_to_string(&args.training_data)?;
        let dataset: TrainingDataset = serde_json::from_str(&training_json)?;

        // Load model
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        varmap.load(&args.model)?;
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = DifficultyModel::new(&vb, 12)?;

        // Normalize features
        let norm_features = dataset.feature_stats.normalize(&features);
        let input = Tensor::from_slice(&norm_features, (1, 12), &device)?;

        // Run inference - CORAL ordinal model produces naturally monotonic outputs
        let (hsk_logits, top_logits) = model.forward(&input)?;

        // Get difficulty scores and learned thresholds for interpretability
        let (hsk_difficulty, top_difficulty, hsk_thresholds, top_thresholds) =
            model.get_difficulty_scores(&input)?;
        let hsk_score = hsk_difficulty.to_vec2::<f32>()?[0][0];
        let top_score = top_difficulty.to_vec2::<f32>()?[0][0];
        let hsk_thresh = hsk_thresholds.to_vec1::<f32>()?;
        let top_thresh = top_thresholds.to_vec1::<f32>()?;

        // Convert to probabilities (naturally monotonic by construction)
        let hsk_probs = candle_nn::ops::sigmoid(&hsk_logits)?.to_vec2::<f32>()?[0].clone();
        let top_probs = candle_nn::ops::sigmoid(&top_logits)?.to_vec2::<f32>()?[0].clone();

        println!("\n  🎯 HSK Difficulty (Neural):");
        for (i, &prob) in hsk_probs.iter().enumerate() {
            let pred = if prob >= 0.5 { "✓" } else { "✗" };
            let level_label = if i < 6 {
                format!("L{}", i + 1)
            } else {
                "L7 (OOV)".to_string()
            };
            println!(
                "    {}: {} (confidence: {:.1}%)",
                level_label,
                pred,
                prob * 100.0
            );
        }

        let hsk_level = hsk_probs.iter().position(|&x| x >= 0.5);
        let hsk_label = match hsk_level {
            Some(i) if i < 6 => format!("HSK level {}", i + 1),
            Some(6) => "HSK level 7 (OOV - very difficult)".to_string(),
            None => "No HSK level (extremely difficult / outside HSK scale)".to_string(),
            _ => "Unknown".to_string(),
        };
        println!("  → Readable from {}", hsk_label);
        println!("  → Difficulty score: {:.2}", hsk_score);
        println!(
            "  → Learned thresholds: [{}]",
            hsk_thresh
                .iter()
                .map(|t| format!("{:.2}", t))
                .collect::<Vec<_>>()
                .join(", ")
        );

        println!("\n  🎯 TOP Difficulty (Neural):");
        for (i, &prob) in top_probs.iter().enumerate() {
            let pred = if prob >= 0.5 { "✓" } else { "✗" };
            let level_label = if i < 4 {
                format!("L{}", i + 1)
            } else {
                "L5 (beyond TOP)".to_string()
            };
            println!(
                "    {}: {} (confidence: {:.1}%)",
                level_label,
                pred,
                prob * 100.0
            );
        }

        let top_level = top_probs.iter().position(|&x| x >= 0.5);
        let top_label = match top_level {
            Some(i) if i < 4 => format!("TOP level {}", i + 1),
            Some(4) => "TOP level 5 (beyond TOP - very difficult)".to_string(),
            None => "No TOP level (extremely difficult / outside TOP scale)".to_string(),
            _ => "Unknown".to_string(),
        };
        println!("  → Readable from {}", top_label);
        println!("  → Difficulty score: {:.2}", top_score);
        println!(
            "  → Learned thresholds: [{}]",
            top_thresh
                .iter()
                .map(|t| format!("{:.2}", t))
                .collect::<Vec<_>>()
                .join(", ")
        );
    } else {
        println!("\n⚠️  Trained model not found. Run `train` first to create a model.");

        // Fallback to rule-based labels
        let (hsk_labels, top_labels) = compute_labels(&matches)?;

        println!("\n🎯 HSK Difficulty (Rule-based):");
        for (i, &label) in hsk_labels.iter().enumerate() {
            let readable = if label == 1 { "✓" } else { "✗" };
            let level_label = if i < 6 {
                format!("L{}", i + 1)
            } else {
                "L7 (OOV)".to_string()
            };
            println!("    {}: {}", level_label, readable);
        }

        let hsk_level = hsk_labels.iter().position(|&x| x == 1);
        let hsk_label = match hsk_level {
            Some(i) if i < 6 => format!("HSK level {}", i + 1),
            Some(6) => "HSK level 7 (OOV - very difficult)".to_string(),
            None => "No HSK level (extremely difficult / outside HSK scale)".to_string(),
            _ => "Unknown".to_string(),
        };
        println!("  → Readable from {}", hsk_label);

        println!("\n🎯 TOP Difficulty (Rule-based):");
        for (i, &label) in top_labels.iter().enumerate() {
            let readable = if label == 1 { "✓" } else { "✗" };
            let level_label = if i < 4 {
                format!("L{}", i + 1)
            } else {
                "L5 (beyond TOP)".to_string()
            };
            println!("    {}: {}", level_label, readable);
        }

        let top_level = top_labels.iter().position(|&x| x == 1);
        let top_label = match top_level {
            Some(i) if i < 4 => format!("TOP level {}", i + 1),
            Some(4) => "TOP level 5 (beyond TOP - very difficult)".to_string(),
            None => "No TOP level (extremely difficult / outside TOP scale)".to_string(),
            _ => "Unknown".to_string(),
        };
        println!("  → Readable from {}", top_label);
    }

    Ok(())
}

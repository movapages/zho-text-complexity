use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "extract_bkrs_phrases")]
#[command(about = "Extract difficult classical phrases from BKRS dictionary")]
struct Args {
    /// Path to BKRS dictionary
    #[arg(long, default_value = "dictionaries/bkrs_dictionary.json")]
    bkrs_dict: PathBuf,

    /// Output file for extracted phrases
    #[arg(short, long, default_value = "data/bkrs_difficult_phrases.json")]
    output: PathBuf,

    /// Minimum phrase length (characters)
    #[arg(long, default_value = "8")]
    min_length: usize,

    /// Maximum phrase length (characters)
    #[arg(long, default_value = "30")]
    max_length: usize,

    /// Maximum number of phrases to extract
    #[arg(long, default_value = "5000")]
    max_count: usize,

    /// Include both simplified and traditional as separate examples
    #[arg(long, default_value = "true")]
    include_traditional: bool,
}

#[derive(Debug, Deserialize)]
struct BkrsEntry {
    sm: String, // simplified
    #[allow(dead_code)]
    tr: Option<String>, // traditional
    #[allow(dead_code)]
    pin: Option<String>, // pinyin
    #[allow(dead_code)]
    lat: Option<String>,
    bkrs: String, // definition/explanation
}

#[derive(Debug, Clone, Serialize)]
struct ExtractedPhrase {
    text: String,
    text_traditional: Option<String>, // Traditional variant if different
    explanation: String,
    category: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("[1/3] Loading BKRS dictionary...");
    let content = fs::read_to_string(&args.bkrs_dict)?;

    println!("[2/3] Parsing and filtering entries...");
    let entries: Vec<BkrsEntry> = serde_json::from_str(&content)?;
    println!("  ✓ Loaded {} total entries", entries.len());

    let mut extracted = Vec::new();
    let mut stats = CategoryStats::default();

    for entry in entries {
        if extracted.len() >= args.max_count {
            break;
        }

        let char_count = entry.sm.chars().count();

        // Filter 1: Length check
        if char_count < args.min_length || char_count > args.max_length {
            continue;
        }

        // Filter 2: Must be mostly Chinese characters
        let chinese_count = entry.sm.chars().filter(|c| is_chinese_char(*c)).count();
        if (chinese_count as f32 / char_count as f32) < 0.8 {
            continue;
        }

        // Classify and filter by category
        if let Some((category, phrase)) = classify_phrase(&entry, &args) {
            // Add simplified version
            extracted.push(phrase.clone());
            stats.increment(&category);

            // Add traditional version if different and requested
            if args.include_traditional {
                if let Some(trad) = &entry.tr {
                    if trad != &entry.sm {
                        // Traditional is different, add it too
                        let trad_phrase = ExtractedPhrase {
                            text: trad.clone(),
                            text_traditional: None, // Mark as the trad version
                            explanation: phrase.explanation.clone(),
                            category: format!("{}_trad", category),
                        };
                        extracted.push(trad_phrase);
                        stats.increment(&format!("{}_trad", category));
                    }
                }
            }

            if extracted.len() % 1000 == 0 {
                println!("  Extracted: {}/{}", extracted.len(), args.max_count);
            }
        }
    }

    println!("  ✓ Extracted {} phrases", extracted.len());

    // Print category breakdown
    println!("\n--- Category Distribution ---");
    println!(
        "  Classical idioms: {} ({}%)",
        stats.classical,
        stats.classical * 100 / extracted.len().max(1)
    );
    println!(
        "  Literary sayings: {} ({}%)",
        stats.literary,
        stats.literary * 100 / extracted.len().max(1)
    );
    println!(
        "  Chengyu (4-char): {} ({}%)",
        stats.chengyu,
        stats.chengyu * 100 / extracted.len().max(1)
    );
    println!(
        "  Complex phrases: {} ({}%)",
        stats.complex,
        stats.complex * 100 / extracted.len().max(1)
    );
    println!(
        "  Traditional variants: {} ({}%)",
        stats.traditional_count,
        stats.traditional_count * 100 / extracted.len().max(1)
    );

    println!("\n[3/3] Saving to {:?}...", args.output);
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(&extracted)?;
    fs::write(&args.output, json)?;

    println!("✓ Saved {} difficult phrases", extracted.len());

    Ok(())
}

fn classify_phrase(entry: &BkrsEntry, args: &Args) -> Option<(String, ExtractedPhrase)> {
    let text = &entry.sm;
    let explanation = &entry.bkrs;
    let char_count = text.chars().count();

    // PERMISSIVE EXTRACTION - let vocabulary matching filter later

    // Category 1: Classical idioms with explanations (HIGH PRIORITY)
    // Look for keywords: 比喻, 形容, 指, 原比喻, 后用作
    if explanation.contains("比喻")
        || explanation.contains("形容")
        || explanation.contains("原比喻")
        || explanation.contains("后用作")
        || explanation.contains("用作")
    {
        if explanation.chars().count() > 10 {
            return Some((
                "classical".to_string(),
                ExtractedPhrase {
                    text: text.clone(),
                    text_traditional: entry.tr.clone(),
                    explanation: explanation.clone(),
                    category: "classical_idiom".to_string(),
                },
            ));
        }
    }

    // Category 2: 4-character chengyu (成语) - VERY HIGH PRIORITY
    if char_count == 4 && text.chars().all(is_chinese_char) {
        if explanation.chars().count() > 15 {
            return Some((
                "chengyu".to_string(),
                ExtractedPhrase {
                    text: text.clone(),
                    text_traditional: entry.tr.clone(),
                    explanation: explanation.clone(),
                    category: "chengyu".to_string(),
                },
            ));
        }
    }

    // Category 3: Literary sayings with commas (classical structure)
    if char_count >= 8 && char_count <= 25 && text.contains("，") {
        if explanation.chars().count() > 10 {
            return Some((
                "literary".to_string(),
                ExtractedPhrase {
                    text: text.clone(),
                    text_traditional: entry.tr.clone(),
                    explanation: explanation.clone(),
                    category: "literary_saying".to_string(),
                },
            ));
        }
    }

    // Category 4: Longer complex phrases (8-30 chars)
    // PERMISSIVE - include everything with substantial explanation
    if char_count >= args.min_length && char_count <= args.max_length {
        if explanation.chars().count() > 20 {
            // Basic filter: mostly Chinese characters
            let chinese_ratio =
                text.chars().filter(|c| is_chinese_char(*c)).count() as f32 / char_count as f32;
            if chinese_ratio > 0.7 {
                return Some((
                    "complex".to_string(),
                    ExtractedPhrase {
                        text: text.clone(),
                        text_traditional: entry.tr.clone(),
                        explanation: explanation.clone(),
                        category: "complex_phrase".to_string(),
                    },
                ));
            }
        }
    }

    None
}

fn is_chinese_char(c: char) -> bool {
    let code = c as u32;
    (code >= 0x4E00 && code <= 0x9FFF)  // CJK Unified Ideographs
        || (code >= 0x3400 && code <= 0x4DBF)  // CJK Extension A
        || (code >= 0xF900 && code <= 0xFAFF) // CJK Compatibility Ideographs
}

#[derive(Default)]
struct CategoryStats {
    classical: usize,
    literary: usize,
    chengyu: usize,
    complex: usize,
    traditional_count: usize,
}

impl CategoryStats {
    fn increment(&mut self, category: &str) {
        // Track traditional variants separately
        if category.ends_with("_trad") {
            self.traditional_count += 1;
        }

        // Base category stats
        if category.starts_with("classical") {
            self.classical += 1;
        } else if category.starts_with("literary") {
            self.literary += 1;
        } else if category.starts_with("chengyu") {
            self.chengyu += 1;
        } else if category.starts_with("complex") {
            self.complex += 1;
        }
    }
}

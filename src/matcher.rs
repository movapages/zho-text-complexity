use crate::WordMatch;
use aho_corasick::AhoCorasick;
use anyhow::Result;
use rustc_hash::FxHashMap;

/// Matcher wrapper combining AC automaton with vocabulary lookups
pub struct Matcher {
    ac: AhoCorasick,
    hsk_vocab: FxHashMap<String, u32>,
    top_vocab: FxHashMap<String, u32>,
}

impl Matcher {
    /// Build a matcher from HSK and TOP vocabularies
    pub fn new(
        hsk_vocab: FxHashMap<String, u32>,
        top_vocab: FxHashMap<String, u32>,
    ) -> Result<Self> {
        // Combine all patterns from both vocabularies
        let mut patterns = Vec::new();
        for word in hsk_vocab.keys() {
            patterns.push(word.as_str());
        }
        for word in top_vocab.keys() {
            if !hsk_vocab.contains_key(word) {
                patterns.push(word.as_str());
            }
        }

        // Build AC automaton
        let ac = AhoCorasick::new(patterns)?;

        Ok(Self {
            ac,
            hsk_vocab,
            top_vocab,
        })
    }

    /// Find all word matches in a sentence
    ///
    /// Returns overlapping matches with their difficulty levels.
    /// Each match includes both byte span and difficulty assignments.
    pub fn find_matches(&self, sentence: &str) -> Vec<WordMatch> {
        let mut matches = Vec::new();

        for mat in self.ac.find_iter(sentence) {
            let text = &sentence[mat.start()..mat.end()];
            let hsk_level = self.hsk_vocab.get(text).copied();
            let top_level = self.top_vocab.get(text).copied();

            matches.push(WordMatch {
                text: text.to_string(),
                hsk_level,
                top_level,
                start: mat.start(),
                end: mat.end(),
            });
        }

        matches
    }

    /// Extract non-overlapping matches using greedy longest-match strategy
    ///
    /// This is useful if you want exactly one match per position.
    /// Returns matches that don't overlap, prioritizing longer matches.
    pub fn find_non_overlapping(&self, sentence: &str) -> Vec<WordMatch> {
        let all_matches = self.find_matches(sentence);
        let mut result = Vec::new();
        let mut covered_until = 0;

        for mat in all_matches {
            if mat.start >= covered_until {
                covered_until = mat.end;
                result.push(mat);
            }
        }

        result
    }

    /// Get vocabulary sizes
    pub fn vocab_sizes(&self) -> (usize, usize) {
        (self.hsk_vocab.len(), self.top_vocab.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab() -> (FxHashMap<String, u32>, FxHashMap<String, u32>) {
        let mut hsk = FxHashMap::default();
        hsk.insert("我".to_string(), 1);
        hsk.insert("喜欢".to_string(), 3);
        hsk.insert("吃".to_string(), 1);
        hsk.insert("苹果".to_string(), 2);
        hsk.insert("和".to_string(), 2);

        let mut top = FxHashMap::default();
        top.insert("我".to_string(), 1);
        top.insert("喜欢".to_string(), 1);
        top.insert("吃".to_string(), 1);
        top.insert("苹果".to_string(), 1);
        top.insert("和".to_string(), 1);

        (hsk, top)
    }

    #[test]
    fn test_matcher_creation() {
        let (hsk, top) = make_vocab();
        let matcher = Matcher::new(hsk, top).unwrap();
        let (hsk_size, top_size) = matcher.vocab_sizes();
        assert!(hsk_size > 0);
        assert!(top_size > 0);
    }

    #[test]
    fn test_find_matches() {
        let (hsk, top) = make_vocab();
        let matcher = Matcher::new(hsk, top).unwrap();

        let sentence = "我喜欢吃苹果。";
        let matches = matcher.find_matches(sentence);

        assert!(!matches.is_empty());
        // Should find: 我, 喜欢, 吃, 苹果
        assert!(matches
            .iter()
            .any(|m| m.text == "我" && m.hsk_level == Some(1)));
        assert!(matches
            .iter()
            .any(|m| m.text == "喜欢" && m.hsk_level == Some(3)));
    }

    #[test]
    fn test_non_overlapping_greedy() {
        let (hsk, top) = make_vocab();
        let matcher = Matcher::new(hsk, top).unwrap();

        let sentence = "我喜欢吃苹果。";
        let non_overlapping = matcher.find_non_overlapping(sentence);

        // All matches should be disjoint
        for i in 0..non_overlapping.len() {
            for j in (i + 1)..non_overlapping.len() {
                let mat_i = &non_overlapping[i];
                let mat_j = &non_overlapping[j];
                // mat_i should end before mat_j starts or vice versa
                assert!(mat_i.end <= mat_j.start || mat_j.end <= mat_i.start);
            }
        }
    }

    #[test]
    fn test_byte_offsets_correct() {
        let (hsk, top) = make_vocab();
        let matcher = Matcher::new(hsk, top).unwrap();

        let sentence = "我喜欢吃苹果。";
        let matches = matcher.find_matches(sentence);

        for mat in matches {
            let extracted = &sentence[mat.start..mat.end];
            assert_eq!(extracted, mat.text);
        }
    }
}

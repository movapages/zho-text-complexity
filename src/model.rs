use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// CORAL (COnsistent RAnk Logits) Ordinal Regression Model
/// Paper: "Deep Neural Networks and Ordinal Regression" (Cao et al.)
///
/// Key features:
/// - Single difficulty score τ per system (HSK/TOP)
/// - Trainable, monotonically-constrained thresholds θ₁ < θ₂ < ... < θₙ
/// - Cumulative probabilities: P(readable at level i) = σ(θᵢ - τ)
/// - Guarantees monotonic predictions by construction
pub struct DifficultyModel {
    shared_fc1: Linear,
    shared_fc2: Linear,
    hsk_difficulty: Linear,     // outputs 1 scalar difficulty score τ
    top_difficulty: Linear,     // outputs 1 scalar difficulty score τ
    hsk_threshold_gaps: Tensor, // K-1 learnable positive gaps for K levels
    top_threshold_gaps: Tensor, // K-1 learnable positive gaps for K levels
}

impl DifficultyModel {
    pub fn new(vs: &VarBuilder, feature_dim: usize) -> Result<Self> {
        // Learnable threshold gaps - initialized to uniform spacing
        // The optimizer will adjust these to find optimal ordinal boundaries
        let hsk_gaps = vs.get_with_hints(6, "hsk.threshold_gaps", candle_nn::init::ZERO)?; // 7 levels → 6 gaps
        let top_gaps = vs.get_with_hints(4, "top.threshold_gaps", candle_nn::init::ZERO)?; // 5 levels → 4 gaps

        Ok(Self {
            shared_fc1: linear(feature_dim, 64, vs.pp("shared.fc1"))?,
            shared_fc2: linear(64, 32, vs.pp("shared.fc2"))?,
            hsk_difficulty: linear(32, 1, vs.pp("hsk.difficulty"))?,
            top_difficulty: linear(32, 1, vs.pp("top.difficulty"))?,
            // K levels → K-1 gaps (NOW TRAINABLE!)
            hsk_threshold_gaps: hsk_gaps, // 7 levels → 6 gaps (includes L7 for OOV)
            top_threshold_gaps: top_gaps, // 5 levels → 4 gaps (TOP has 1-4 + L5 for beyond-TOP)
        })
    }

    /// Compute monotonically increasing thresholds from learned gaps
    /// θ₁ = gap₁, θ₂ = θ₁ + gap₂, θ₃ = θ₂ + gap₃, ...
    /// Ensures θ₁ < θ₂ < θ₃ < ... < θₙ
    ///
    /// CRITICAL: Must use pure tensor operations (no to_vec1!) to maintain gradient flow
    fn compute_thresholds(&self, gaps: &Tensor) -> Result<Tensor> {
        let device = gaps.device();

        // Apply softplus to ensure all gaps are positive: gap = ln(1 + exp(x)) ≥ 0
        // softplus(x) = log(1 + exp(x))
        let exp_gaps = gaps.exp()?;
        let one = Tensor::ones_like(gaps)?;
        let positive_gaps = (one + exp_gaps)?.log()?; // shape: (K-1,)

        // Prepend the base threshold (1.0) to get K elements for K levels
        let base = Tensor::new(&[1.0f32], device)?; // (1,)
        let gaps_with_base = Tensor::cat(&[&base, &positive_gaps], 0)?; // (K,)

        // Cumulative sum to get monotonic thresholds
        // This keeps everything in tensor-land so gradients flow back to gaps
        let thresholds = gaps_with_base.cumsum(0)?; // (K,)

        Ok(thresholds)
    }

    /// Forward pass returns cumulative logits for each level
    /// Formula: logits[i] = θᵢ - τ
    /// where θᵢ are learned monotonic thresholds and τ is predicted difficulty
    ///
    /// Guarantees: P(readable at level i) ≤ P(readable at level i+1)
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let batch_size = x.dims()[0];
        let h = self.shared_fc1.forward(x)?.relu()?;
        let h = self.shared_fc2.forward(&h)?.relu()?;

        // Get difficulty scores: (batch_size, 1)
        let hsk_tau = self.hsk_difficulty.forward(&h)?;
        let top_tau = self.top_difficulty.forward(&h)?;

        // Compute monotonic thresholds from learned gaps
        let hsk_thresholds = self.compute_thresholds(&self.hsk_threshold_gaps)?; // 7 HSK levels (includes L7 for OOV)
        let top_thresholds = self.compute_thresholds(&self.top_threshold_gaps)?; // 5 TOP levels (1-4 + L5 for beyond-TOP)

        // Broadcast thresholds to batch: (K,) → (batch_size, K)
        let hsk_thresholds = hsk_thresholds.unsqueeze(0)?.broadcast_as((batch_size, 7))?;
        let top_thresholds = top_thresholds.unsqueeze(0)?.broadcast_as((batch_size, 5))?;

        // Compute cumulative logits: θᵢ - τ
        // Higher threshold → higher level → needs easier text (lower difficulty)
        let hsk_logits = hsk_thresholds.broadcast_sub(&hsk_tau)?;
        let top_logits = top_thresholds.broadcast_sub(&top_tau)?;

        Ok((hsk_logits, top_logits))
    }

    /// Get raw difficulty scores and learned thresholds for interpretability
    /// Returns: (difficulty, thresholds) for HSK and TOP
    pub fn get_difficulty_scores(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let h = self.shared_fc1.forward(x)?.relu()?;
        let h = self.shared_fc2.forward(&h)?.relu()?;

        let hsk_tau = self.hsk_difficulty.forward(&h)?;
        let top_tau = self.top_difficulty.forward(&h)?;

        // Get learned thresholds
        let hsk_thresholds = self.compute_thresholds(&self.hsk_threshold_gaps)?;
        let top_thresholds = self.compute_thresholds(&self.top_threshold_gaps)?;

        Ok((hsk_tau, top_tau, hsk_thresholds, top_thresholds))
    }
}

/// CORAL ordinal cross-entropy loss
/// Uses binary cross-entropy on cumulative labels
/// Equivalent to: sum_i BCE(σ(θᵢ - τ), label[i])
pub fn binary_cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let targets = targets.to_dtype(DType::F32)?;
    let probs = candle_nn::ops::sigmoid(logits)?;
    let probs = probs.clamp(1e-7, 1.0 - 1e-7)?;

    let log_p = probs.log()?;
    let pos = (&targets * &log_p)?;

    let one = Tensor::ones_like(&probs)?;
    let one_minus_p = (one - &probs)?.log()?;
    let one_minus_t = (Tensor::ones_like(&targets)? - &targets)?;
    let neg = (&one_minus_t * &one_minus_p)?;

    Ok((pos + neg)?.neg()?.mean_all()?)
}

pub fn binary_accuracy(logits: &Tensor, targets: &Tensor, thr: f32) -> Result<Tensor> {
    let preds = candle_nn::ops::sigmoid(logits)?.ge(thr)?;
    let targets_bool = targets.to_dtype(DType::F32)?.ge(0.5)?;
    let correct = preds.eq(&targets_bool)?;
    correct.to_dtype(DType::F32)?.mean_all()
}

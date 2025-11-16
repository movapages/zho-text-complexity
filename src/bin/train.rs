use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use clap::Parser;
use serde_json;
use std::fs;
use std::path::PathBuf;

use zho_complexity::{
    binary_accuracy, binary_cross_entropy, DifficultyModel, FeatureStats, TrainingDataset,
    TrainingExample,
};

#[derive(Parser, Debug)]
#[command(name = "train")]
struct Args {
    #[arg(short, long, default_value = "data/training_data.json")]
    data: PathBuf,

    #[arg(short, long, default_value = "data/model.safetensors")]
    model: PathBuf,

    #[arg(long, default_value = "20")]
    epochs: usize,

    #[arg(long, default_value = "32")]
    batch_size: usize,

    #[arg(long, default_value = "0.001")]
    lr: f64,

    #[arg(long, default_value = "0.2")]
    val_ratio: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("\n[1/5] Loading dataset...");
    let raw = fs::read_to_string(&args.data)?;
    let dataset: TrainingDataset = serde_json::from_str(&raw)?;
    println!("  Loaded {} examples", dataset.examples.len());

    let val_n = (dataset.examples.len() as f32 * args.val_ratio) as usize;
    let train_n = dataset.examples.len() - val_n;
    let (train_ex, val_ex) = dataset.examples.split_at(train_n);

    println!("  Train: {} | Val: {}", train_ex.len(), val_ex.len());

    println!("\n[2/5] Initializing model...");
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("  Using device: {:?}", device);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);

    let model = DifficultyModel::new(&vb, 12)?;
    println!("  Model created");

    println!("\n[3/5] Initializing optimizer...");
    let cfg = ParamsAdam {
        lr: args.lr,
        ..Default::default()
    };
    let mut opt = Adam::new(varmap.all_vars(), cfg)?;
    println!("  Adam optimizer ready (lr={})", args.lr);

    println!("\n[4/5] Training...");
    let mut best_val = f32::INFINITY;

    for epoch in 0..args.epochs {
        let (train_loss, _train_acc) = train_epoch(
            train_ex,
            &model,
            &dataset.feature_stats,
            args.batch_size,
            &device,
            &mut opt,
        )?;

        let (val_loss, _val_acc) = eval_epoch(
            val_ex,
            &model,
            &dataset.feature_stats,
            args.batch_size,
            &device,
        )?;

        let mark = if val_loss < best_val {
            best_val = val_loss;
            "✓"
        } else {
            " "
        };

        // Get separate accuracies for diagnostics
        let (train_hsk_acc, train_top_acc) = get_separate_accs(
            train_ex,
            &model,
            &dataset.feature_stats,
            args.batch_size,
            &device,
        )?;
        let (val_hsk_acc, val_top_acc) = get_separate_accs(
            val_ex,
            &model,
            &dataset.feature_stats,
            args.batch_size,
            &device,
        )?;

        println!(
            "Epoch {:02} | Train L {:.4} (HSK:{:.3} TOP:{:.3}) | Val L {:.4} (HSK:{:.3} TOP:{:.3}) {}",
            epoch + 1,
            train_loss,
            train_hsk_acc,
            train_top_acc,
            val_loss,
            val_hsk_acc,
            val_top_acc,
            mark
        );
    }

    println!("\n[5/5] Saving model...");
    varmap.save(&args.model)?;
    println!("  ✓ Model saved to {:?}", args.model);

    Ok(())
}

fn train_epoch(
    batch: &[TrainingExample],
    model: &DifficultyModel,
    stats: &FeatureStats,
    bs: usize,
    device: &Device,
    opt: &mut Adam,
) -> Result<(f32, f32)> {
    let mut total_loss = 0.0;
    let mut total_acc = 0.0;
    let mut count = 0;

    for chunk in batch.chunks(bs) {
        let (x, hsk_t, top_t) = prepare_batch(chunk, stats, device)?;

        let (hsk_logits, top_logits) = model.forward(&x)?;

        let hsk_loss = binary_cross_entropy(&hsk_logits, &hsk_t)?;
        let top_loss = binary_cross_entropy(&top_logits, &top_t)?;
        let loss = hsk_loss
            .broadcast_add(&top_loss)?
            .broadcast_mul(&Tensor::new(0.5f32, device)?)?;

        opt.backward_step(&loss)?;

        total_loss += loss.to_vec0::<f32>()?;

        let hsk_acc = binary_accuracy(&hsk_logits, &hsk_t, 0.5)?.to_vec0::<f32>()?;
        let top_acc = binary_accuracy(&top_logits, &top_t, 0.5)?.to_vec0::<f32>()?;
        total_acc += (hsk_acc + top_acc) * 0.5;

        count += 1;
    }

    Ok((total_loss / count as f32, total_acc / count as f32))
}

fn eval_epoch(
    batch: &[TrainingExample],
    model: &DifficultyModel,
    stats: &FeatureStats,
    bs: usize,
    device: &Device,
) -> Result<(f32, f32)> {
    let mut total_loss = 0.0;
    let mut total_acc = 0.0;
    let mut count = 0;

    for chunk in batch.chunks(bs) {
        let (x, hsk_t, top_t) = prepare_batch(chunk, stats, device)?;

        let (hsk_logits, top_logits) = model.forward(&x)?;

        let hsk_loss = binary_cross_entropy(&hsk_logits, &hsk_t)?;
        let top_loss = binary_cross_entropy(&top_logits, &top_t)?;
        let loss = hsk_loss
            .broadcast_add(&top_loss)?
            .broadcast_mul(&Tensor::new(0.5f32, device)?)?;

        total_loss += loss.to_vec0::<f32>()?;

        let hsk_acc = binary_accuracy(&hsk_logits, &hsk_t, 0.5)?.to_vec0::<f32>()?;
        let top_acc = binary_accuracy(&top_logits, &top_t, 0.5)?.to_vec0::<f32>()?;
        total_acc += (hsk_acc + top_acc) * 0.5;

        count += 1;
    }

    Ok((total_loss / count as f32, total_acc / count as f32))
}

fn get_separate_accs(
    batch: &[TrainingExample],
    model: &DifficultyModel,
    stats: &FeatureStats,
    bs: usize,
    device: &Device,
) -> Result<(f32, f32)> {
    let mut hsk_total = 0.0;
    let mut top_total = 0.0;
    let mut count = 0;

    for chunk in batch.chunks(bs) {
        let (x, hsk_t, top_t) = prepare_batch(chunk, stats, device)?;
        let (hsk_logits, top_logits) = model.forward(&x)?;

        let hsk_acc = binary_accuracy(&hsk_logits, &hsk_t, 0.5)?.to_vec0::<f32>()?;
        let top_acc = binary_accuracy(&top_logits, &top_t, 0.5)?.to_vec0::<f32>()?;

        hsk_total += hsk_acc;
        top_total += top_acc;
        count += 1;
    }

    Ok((hsk_total / count as f32, top_total / count as f32))
}

/// Prepare tensors using `from_slice`
fn prepare_batch(
    exs: &[TrainingExample],
    stats: &FeatureStats,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let n = exs.len();

    let mut f = Vec::new();
    for e in exs {
        f.extend(stats.normalize(&e.features));
    }
    let x = Tensor::from_slice(&f, (n, 12), device)?;

    let hsk_flat: Vec<f32> = exs
        .iter()
        .flat_map(|e| e.hsk_labels.iter().map(|&v| v as f32))
        .collect();
    let hsk = Tensor::from_slice(&hsk_flat, (n, 7), device)?;

    let top_flat: Vec<f32> = exs
        .iter()
        .flat_map(|e| e.top_labels.iter().map(|&v| v as f32))
        .collect();
    let top = Tensor::from_slice(&top_flat, (n, 5), device)?; // TOP has 5 levels (1-4 + beyond)

    Ok((x, hsk, top))
}

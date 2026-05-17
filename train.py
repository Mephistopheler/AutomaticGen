from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.aqg.transformers_text import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)

from src.aqg.config_utils import load_config, save_json
from src.aqg.data import load_dataset_group, tokenize_dataset
from src.aqg.metrics import compute_text_metrics
from src.aqg.utils import ensure_dir, format_seconds, get_device, set_seed


def get_autocast_context(device: torch.device, mixed_precision: str):
    if device.type != 'cuda' or mixed_precision == 'no':
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == 'bf16' else torch.float16
    return torch.amp.autocast(device_type='cuda', dtype=dtype)


def resolve_mixed_precision(value: Any, device: torch.device) -> str:
    mode = str(value or 'no').lower()
    if mode in {'false', 'none', 'off'}:
        return 'no'
    if mode in {'true', 'yes', 'auto'}:
        if device.type == 'cuda':
            return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
        return 'no'
    if mode not in {'no', 'fp16', 'bf16'}:
        raise ValueError(f'Unsupported mixed_precision value: {value!r}')
    if device.type != 'cuda':
        return 'no'
    if mode == 'bf16' and not torch.cuda.is_bf16_supported():
        print('bf16 is not supported on this GPU; falling back to fp16.')
        return 'fp16'
    return mode


def save_model_checkpoint(model: Any, tokenizer: Any, output_dir: Path) -> None:
    previous_use_cache = getattr(getattr(model, 'config', None), 'use_cache', None)
    if previous_use_cache is not None:
        model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if previous_use_cache is not None:
        model.config.use_cache = previous_use_cache


def evaluate_loss(model: Any, dataloader: DataLoader, device: torch.device, mixed_precision: str = 'no') -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with get_autocast_context(device, mixed_precision):
                outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            total_steps += 1
    return total_loss / max(total_steps, 1)


def generate_predictions(
    model: Any,
    tokenizer: Any,
    dataset,
    device: torch.device,
    batch_size: int,
    generation_cfg: Dict[str, Any],
) -> List[str]:
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    predictions: List[str] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating', leave=False):
            batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            generated = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=generation_cfg.get('max_new_tokens', 64),
                num_beams=generation_cfg.get('num_beams', 4),
                do_sample=generation_cfg.get('do_sample', False),
                temperature=generation_cfg.get('temperature', 1.0),
                top_p=generation_cfg.get('top_p', 1.0),
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend([text.strip() for text in decoded])
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description='Train AQG seq2seq model')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg['seed']))

    output_dir = ensure_dir(cfg['output_dir'])
    checkpoint_dir = ensure_dir(output_dir / 'checkpoints')
    logs_dir = ensure_dir(output_dir / 'logs')

    device = get_device()
    print(f'Using device: {device}')

    model_cfg = dict(cfg['model'])
    model_name = model_cfg.pop('name')
    tokenizer_use_fast = bool(model_cfg.pop('use_fast', False))
    use_safetensors = bool(model_cfg.pop('use_safetensors', True))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=tokenizer_use_fast)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_safetensors=use_safetensors,
            **model_cfg,
        )
    except ValueError as exc:
        if 'torch to at least v2.6' in str(exc):
            raise RuntimeError(
                'Model loading failed because your Torch version is below 2.6 and '
                '`transformers` now blocks `torch.load` for security reasons '                '(CVE-2025-32434). '                'Upgrade Torch to >=2.6 OR set model.use_safetensors=true and use a '                'checkpoint that provides safetensors weights.'
            ) from exc
        raise
    except OSError as exc:
        if use_safetensors:
            raise RuntimeError(
                f'Model loading failed for {model_name!r} while use_safetensors=true. '
                'This checkpoint may not provide safetensors weights, and the automatic '
                'Hugging Face conversion service did not return a usable response. '
                'Set model.use_safetensors=false to load the PyTorch checkpoint instead.'
            ) from exc
        raise
    model.to(device)

    training_cfg = cfg.get('training', {})
    mixed_precision = resolve_mixed_precision(training_cfg.get('mixed_precision', 'no'), device)
    if training_cfg.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        if hasattr(model, 'config'):
            model.config.use_cache = False
        print('Gradient checkpointing enabled.')
    print(f'Mixed precision: {mixed_precision}')

    template = cfg['data']['source_template']
    train_dataset_raw = load_dataset_group(cfg['data']['train_datasets'], template=template)
    val_dataset_raw = load_dataset_group(cfg['data']['validation_datasets'], template=template)

    train_dataset = tokenize_dataset(
        train_dataset_raw,
        tokenizer=tokenizer,
        max_source_length=int(cfg['data']['max_source_length']),
        max_target_length=int(cfg['data']['max_target_length']),
    )
    val_dataset = tokenize_dataset(
        val_dataset_raw,
        tokenizer=tokenizer,
        max_source_length=int(cfg['data']['max_source_length']),
        max_target_length=int(cfg['data']['max_target_length']),
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    batch_size = int(cfg['training']['batch_size'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=float(training_cfg['learning_rate']))
    epochs = int(training_cfg['num_epochs'])
    grad_accum_steps = int(training_cfg.get('gradient_accumulation_steps', 1))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and mixed_precision == 'fp16'))
    save_each_epoch = bool(training_cfg.get('save_each_epoch', True))
    total_training_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    warmup_ratio = float(training_cfg.get('warmup_ratio', 0.1))
    num_warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    training_log: List[Dict[str, Any]] = []
    best_val_loss = float('inf')
    patience = int(training_cfg.get('early_stopping_patience', 2))
    patience_counter = 0
    global_step = 0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with get_autocast_context(device, mixed_precision):
                outputs = model(**batch)
                batch_loss = outputs.loss
                loss = batch_loss / grad_accum_steps
            scaler.scale(loss).backward()
            epoch_loss += float(batch_loss.item())

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            progress.set_postfix(loss=f'{loss.item() * grad_accum_steps:.4f}')

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        avg_val_loss = evaluate_loss(model, val_loader, device, mixed_precision=mixed_precision)

        epoch_record = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'global_step': global_step,
        }
        training_log.append(epoch_record)
        print(epoch_record)

        if save_each_epoch:
            epoch_dir = ensure_dir(checkpoint_dir / f'epoch_{epoch}')
            save_model_checkpoint(model, tokenizer, epoch_dir)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_dir = ensure_dir(checkpoint_dir / 'best')
            save_model_checkpoint(model, tokenizer, best_dir)
            print(f'Best checkpoint updated: {best_dir}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered.')
                break

    elapsed = time.time() - start_time
    save_json(
        {
            'config_path': str(Path(args.config).resolve()),
            'training_log': training_log,
            'best_val_loss': best_val_loss,
            'elapsed_seconds': elapsed,
            'elapsed_hms': format_seconds(elapsed),
            'device': str(device),
            'mixed_precision': mixed_precision,
            'gradient_checkpointing': bool(training_cfg.get('gradient_checkpointing', False)),
            'save_each_epoch': save_each_epoch,
        },
        logs_dir / 'training_summary.json',
    )

    if cfg.get('run_validation_generation', True):
        best_dir = checkpoint_dir / 'best'
        model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(best_dir, use_fast=tokenizer_use_fast)
        predictions = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            dataset=val_dataset,
            device=device,
            batch_size=batch_size,
            generation_cfg=cfg['generation'],
        )
        references = [row['target'] for row in val_dataset_raw]
        metrics = compute_text_metrics(
            predictions=predictions,
            references=references,
            bertscore_model_type=cfg['evaluation'].get('bertscore_model_type'),
            language=cfg['evaluation'].get('language', 'ru'),
        )
        save_json(metrics, logs_dir / 'validation_metrics.json')
        preview_rows = []
        for row, pred in zip(val_dataset_raw, predictions):
            preview_rows.append(
                {
                    'example_id': row['example_id'],
                    'dataset_name': row['dataset_name'],
                    'context': row['context'],
                    'answer_text': row['answer_text'],
                    'reference_question': row['target'],
                    'generated_question': pred,
                }
            )
        from src.aqg.utils import save_jsonl

        save_jsonl(preview_rows, logs_dir / 'validation_predictions.jsonl')
        print('Validation metrics:')
        print(metrics)


if __name__ == '__main__':
    main()

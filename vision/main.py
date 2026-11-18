#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import time


def _parse_poly_hidden_dims(raw: str):
    """Parse comma-separated hidden dimensions into a list of ints."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def _parse_float_list(raw: str):
    """Parse a comma-separated float list ('' -> None)."""
    if raw is None or not raw.strip():
        return None
    return [float(x.strip()) for x in raw.split(',') if x.strip()]


def _parse_int_list(raw: str):
    """Parse a comma-separated int list ('' -> None)."""
    if raw is None or not raw.strip():
        return None
    return [int(x.strip()) for x in raw.split(',') if x.strip()]

def main():
    parser = argparse.ArgumentParser(
        description="Train a sparse autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Feature LMDBs. Either pass the three paths explicitly, or let them be resolved from
    # --features_root + --dataset + --extractor (layout: <root>/<dataset>/<split>/<file>.lmdb,
    # as written by extract_imagenet.py / extract_open_images.py).
    parser.add_argument('--train_path', default=None)
    parser.add_argument('--val_path', default=None)
    parser.add_argument('--test_path', default=None)
    parser.add_argument('--features_root', type=str,
                       default=os.environ.get('FEATURES_ROOT', './features'),
                       help='Root of the extracted feature LMDBs (env FEATURES_ROOT); only used '
                            'for paths not given explicitly')
    
    # Model selection
    parser.add_argument('--model', type=str, default='topk',
                       choices=['topk', 'batch_topk', 'vanilla', 'jumprelu', 'gated', 'dual',
                                'matryoshka', 'msae'],
                       help='Model architecture: topk (TopKSAE), batch_topk (BatchTopKSAE), '
                            'vanilla (VanillaSAE), jumprelu (JumpReLUSAE), '
                            'gated (Rajamanoharan et al. GatedSAE, arXiv 2404.16014 — '
                            'EMERGENT L0, tune --l1_coef against a target L0), '
                            'dual (DualSAE), '
                            'matryoshka (Bussmann dictionary-prefix MatryoshkaSAE), '
                            'or msae (Zaigrajew sparsity-level-nesting MSAE)')
    # The non-vision dataset names (text corpora, language-model residual streams, VLM token
    # shards) are metadata only: they are stored as dataset_name in training_config.json and
    # never read for computation. They have no built-in path table, so all three --*_path flags
    # must be passed.
    parser.add_argument('--dataset', type=str,
                       choices=['imagenet', 'open_images',
                                'dbpedia14', 'agnews', 'news20', 'goemotions', 'pythia410m',
                                'vlmtok'],
                       default='imagenet',
                       help='Dataset to train/evaluate on. The vision names resolve a built-in '
                            'path table; the language names carry no table and REQUIRE explicit '
                            '--train_path/--val_path/--test_path.')
    # 'clipL336L22' / 'ivit300mL23' name VLM vision-tower token sites; like the text extractors
    # they are metadata only (stored as extractor_name), never read for computation.
    parser.add_argument('--extractor', type=str,
                       choices=['clip', 'vit', 'sigclip', 'dino', 'mpnet', 'bge', 'e5', 'resid',
                                'clipL336L22', 'ivit300mL23'],
                       default='clip',
                       help='Feature extractor to use')

    # Model architecture
    parser.add_argument('--n_latents', type=int, default=8192, 
                       help='Number of latent dimensions')
    parser.add_argument('--activation', type=str, default='topk', 
                       choices=['relu', 'topk'],
                       help='Activation function type')
    parser.add_argument('--topk_k', type=int, default=64, 
                       help='K for TopK activation')
    parser.add_argument('--tied_weights', action='store_true', default=True,
                       help='Use tied encoder/decoder weights')
    # --tied_weights is store_true with default=True, so it can never turn tying OFF; it is kept
    # exactly as it was so every existing command line still parses to the same value.  The
    # untied TopK baseline needs a way to say
    # False, hence this companion flag writing the SAME dest.  argparse applies the first-declared
    # default (True), so omitting both flags is unchanged; if both are given the last one wins.
    # TopK (--model topk) is the only trainer that reads it -- validated below.
    parser.add_argument('--no_tied_weights', dest='tied_weights', action='store_false',
                       help='Untie the TopK encoder/decoder (--model topk only). The decoder is '
                            'then unit-normed with gradient projection every step and '
                            're-initialised with the encoder on dead-latent reinit.')
    parser.add_argument('--normalize', action='store_true', default=False,
                       help='Normalize inputs to zero mean, unit variance')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2048, 
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--eps', type=float, default=6.25e-10,
                       help='Epsilon for Adam optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Loss coefficients
    parser.add_argument('--auxk_coef', type=float, default=1/32, 
                       help='Coefficient for auxiliary TopK loss')
    parser.add_argument('--auxk_k', type=int, default=None,
                       help='K for auxiliary TopK (if None: topk uses topk_k, batch_topk uses '
                            '512)')
    parser.add_argument('--l1_coef', type=float, default=0.0001,
                       help='Coefficient for L1 sparsity (vanilla) or L0 sparsity (jumprelu)')
    parser.add_argument('--bandwidth', type=float, default=0.001,
                       help='Bandwidth for JumpReLU activation (smoothness of threshold)')
    # JumpReLU threshold initialization (theta itself, not log theta; must be > 0). Unset means
    # "use the model default", theta = 1.0, which is the initialization of the vision runs; the
    # key is then left out of the model config and of training_config.json. Families trained
    # with another value pass it explicitly (e.g. --jump_threshold_init 0.001).
    parser.add_argument('--jump_threshold_init', type=float, default=None,
                       help='Initial JumpReLU threshold theta for --model jumprelu and for '
                            '--model dual --sparsity_mode jump_relu. Unset = model default '
                            '(theta = 1.0).')
    # Opt-in deviation from the default Dual-JumpReLU recipe. By default train_dual rebuilds the
    # optimizer at the end of warm-up WITHOUT jumprelu.log_threshold (POST_WARMUP_EXCLUDED_PARAMS),
    # so the Dual-JumpReLU threshold trains during warm-up only, while the JumpReLU baseline
    # trains its threshold for all steps. This flag keeps the threshold in the post-warm-up
    # optimizer (mono group, mono lr). Default False; a False value is dropped from
    # training_config.json (utils.config_payload).
    parser.add_argument('--dual_train_threshold_post_warmup', action='store_true', default=False,
                       help='--model dual --sparsity_mode jump_relu only: keep training the '
                            'JumpReLU threshold after the poly warm-up (the default recipe '
                            'freezes it). Default off.')
    parser.add_argument('--mono_coef', type=float, default=0.0,
                       help='Coefficient for monosemanticity loss (1 - mono_score)')
    parser.add_argument('--mono_period', type=int, default=1,
                       help='Compute mono loss every N steps (1=per batch, >1=full dataset every N steps)')
    
    # Warm start (DualSAE only)
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Train mono branch alone for this many steps (DualSAE only, 0=disabled)')
    parser.add_argument('--warmup_mono_target', type=float, default=0.0,
                       help='Train mono branch until mono score reaches this value (DualSAE only, 0=disabled)')
    parser.add_argument('--warmup_check_interval', type=int, default=500,
                       help='Check mono score every N steps during warmup (DualSAE only)')
    parser.add_argument('--poly_lr_factor', type=float, default=1.0,
                       help='Learning rate multiplier for poly branch after warmup (DualSAE only, e.g., 0.1 = 10%% of base lr)')
    
    # DualSAE specific
    parser.add_argument('--poly_hidden', type=int, default=512,
                       help='Hidden dimension for polynomial MLP (DualSAE only)')
    parser.add_argument('--poly_layers', type=int, default=1,
                       help='Number of linear layers in polynomial projector MLP (DualSAE only)')
    parser.add_argument('--poly_hidden_dims', type=str, default='',
                       help='Comma-separated hidden widths for projector MLP, e.g. 1024,512,256 (overrides poly_layers/poly_hidden)')
    # Residual-branch ablation options. All default to the standard recipe; train_dual's
    # TrainingConfig explains each one, and utils.config_payload drops them when unset.
    parser.add_argument('--poly_arch', type=str, default='mlp', choices=['mlp', 'none'],
                       help="DualSAE only: 'none' removes the residual branch, giving a standard "
                            'SAE that runs the exact Dual sparse-branch code (needs no warm-up, '
                            'span 0)')
    parser.add_argument('--poly_final_act', type=str, default='auto',
                       choices=['auto', 'none', 'gelu'],
                       help="DualSAE only: projector output activation; 'auto' keeps each legacy "
                            'path (1-layer ends in GELU, deeper paths end in Linear)')
    parser.add_argument('--poly_warmup_zero', action='store_true', default=False,
                       help='DualSAE only: hold r_hat at 0 during warm-up, so warm-up is a plain '
                            'SAE objective instead of including the frozen random projector')
    parser.add_argument('--freeze_sparse_from', type=str, default=None,
                       help="DualSAE only: finished --poly_arch none run whose sparse branch is "
                            'frozen; only the projector trains here')
    parser.add_argument('--snapshot_steps', type=str, default='',
                       help='With --freeze_sparse_from: comma-separated optimizer steps at which '
                            'to also save a loadable <exp_name>__step<T>/ directory')
    parser.add_argument('--sparsity_mode', type=str, default='topk',
                       choices=['topk', 'batch_topk', 'vanilla', 'jump_relu',
                                'matryoshka', 'msae', 'gated'],
                       help='Sparsity mode for DualSAE: topk, batch_topk, vanilla, jump_relu, '
                            'matryoshka (Dual-Matryoshka), msae (Dual-MSAE), or '
                            'gated (Dual-Gated; EMERGENT L0, tune --l1_coef)')
    # Paper name: decorrelation loss L_dcor with coefficient gamma. --decorrelation_coef is an
    # alias; the value is stored as span_coef everywhere (configs, checkpoints).
    parser.add_argument('--span_coef', '--decorrelation_coef', dest='span_coef', type=float,
                       default=0.01,
                       help='DualSAE only: coefficient gamma of the decorrelation loss L_dcor '
                            '(span-overlap penalty between r_hat and the active decoder atoms)')
    parser.add_argument('--sim_coef', type=float, default=0.0,
                       help='Coefficient for similarity penalty between xM and r_hat (DualSAE only)')
    parser.add_argument('--active_only_span', '--active_only_decorrelation', dest='active_only_span',
                       action='store_true', default=True,
                       help='Only apply span penalty to active latents (DualSAE only)')
    parser.add_argument('--save_val_embeds', action='store_true', default=False,
                       help='Save validation embeddings during training (DualSAE only)')
    parser.add_argument('--save_embeds_interval', type=int, default=1,
                       help='Save validation embeddings every N epochs (DualSAE only)')

    # Gated-SAE knob (--model gated / --sparsity_mode gated only; inert otherwise)
    parser.add_argument('--allow_degenerate_warmup', action='store_true', default=False,
                       help='DualSAE: permit warmup_steps >= the total step budget (the '
                            'Residual Branch then never unfreezes and the run is a plain SAE). '
                            'E.g. OpenImagesV7 at batch 2048 has 247 steps/epoch, i.e. 12,350 '
                            'steps in 50 epochs, so warm-up 17500 needs this flag.')
    parser.add_argument('--gated_sparsity_warmup', type=int, default=2000,
                       help='GatedSAE: linear ramp of the gate-path L1 over this many '
                            'optimizer steps (0 disables). Reference default 2000.')

    # Matryoshka-family knobs (matryoshka / msae models and sparsity modes)
    parser.add_argument('--group_fractions', type=str,
                       default='0.03125,0.0625,0.125,0.25,0.53125',
                       help='Matryoshka (Bussmann) dictionary-group fractions, comma-separated, '
                            'summing to 1; sizes = int(f*n_latents) with the remainder in the '
                            'last group (upstream dictionary_learning convention: '
                            '[1/32,1/16,1/8,1/4,1/2+1/32])')
    parser.add_argument('--msae_nesting', type=str, default='',
                       help='MSAE sparsity-level nesting, comma-separated ks (empty = official '
                            'construction build_nesting(topk_k, 256), e.g. 64,128,256)')
    parser.add_argument('--msae_weighting', type=str, default='rw', choices=['rw', 'uw'],
                       help='MSAE per-level loss weights: rw (reverse, highest weight on the '
                            'smallest/deployed k — reference config) or uw (uniform)')
    parser.add_argument('--msae_soft_cap', type=float, default=30.0,
                       help='MSAE SoftCapping value cap*tanh(x/cap); <=0 disables')

    # --- non-vision flags (every default leaves the vision runs unchanged) ---------------------
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                       help='Fix the optimizer-step budget per epoch instead of taking one full '
                            'pass over the loader. Batches come from ONE persistent iterator, so '
                            'a small corpus advances through the data across epochs rather than '
                            'replaying the same prefix. Unset (the default) = unchanged.')
    parser.add_argument('--multilabel', dest='multilabel', action='store_true', default=None,
                       help='Declare the targets multi-hot. Recorded in training_config.json so a '
                            'later re-score cannot disagree with the run. Unset = infer from the '
                            "LMDB's __multilabel__ marker, then the legacy path substring.")
    parser.add_argument('--no_multilabel', dest='multilabel', action='store_false',
                       help='Declare the targets single-label (see --multilabel).')
    parser.add_argument('--labels_lmdb', type=str, default=None,
                       help='Absolute path to a companion multi-hot label store shared by '
                            'several feature LMDBs (e.g. one per text encoder).')
    parser.add_argument('--force_normalize', type=str, default='auto',
                       choices=['auto', 'on', 'off'],
                       help="Control input_unit_norm (per-sample standardization). 'auto' keeps "
                            "the historical behaviour, which hard-codes True for --model topk and "
                            "otherwise honours --normalize. 'on'/'off' override it, which the "
                            "language cells need because normalize is a function of (activation, "
                            "variant) and, for a residual stream, must be off.")

    # Eval/IO stripping (wired for all trainers; training math is unchanged)
    parser.add_argument('--skip_extra_eval', action='store_true', default=False,
                       help='Skip the heavy logging-only evals (per-epoch train-set R2/mono, '
                            'end-of-run KNN/LP, intermediates). Best-val selection and all '
                            'checkpoints are still written; downstream re-scores everything.')
    parser.add_argument('--skip_epoch_checkpoints', action='store_true', default=False,
                       help='DualSAE only: do not write the ~150MB per-epoch checkpoints under '
                            'epoch_checkpoints/ (nothing in this repository reads them)')
    parser.add_argument('--select_from_epoch', type=int, default=0,
                       help='DualSAE trainer only: the best-val-MonoScore checkpoint is chosen '
                            'among epochs >= this (1-based); 0 = any epoch (default)')
    
    # Dead neuron handling
    parser.add_argument('--dead_steps_threshold', type=int, default=10_000_000, 
                       help='Steps threshold for considering a neuron dead')
    parser.add_argument('--dead_check_interval', type=int, default=1000,
                       help='Check for dead neurons every N steps (topk model only)')
    
    # Logging
    # W&B is opt-in (and imported only when this is set). TensorBoard (--tb_dir) and the
    # best_metrics.json file written next to every checkpoint need no network access.
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Wandb project name (unset: no wandb logging)')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group name for organizing runs')
    
    # System
    parser.add_argument('--exp_name', type=str,
                       help='Experiment name (checkpoints saved to output_dir/exp_name)')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Base output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                       help='Skip training if final checkpoint already exists')
    parser.add_argument('--tb_dir', type=str, default=None,
                       help='TensorBoard event dir (real-time, offline); None disables TB logging')

    args = parser.parse_args()

    # Validate the two opt-in flags against the architecture that would actually read them.
    # Without this, --jump_threshold_init on a non-JumpReLU model and --no_tied_weights on any
    # model but topk would be accepted and SILENTLY IGNORED (the other trainers never look at
    # them), producing a run whose name/provenance claims a setting it did not have.
    if args.jump_threshold_init is not None:
        import math
        is_jumprelu = (args.model == 'jumprelu'
                       or (args.model == 'dual' and args.sparsity_mode == 'jump_relu'))
        if not is_jumprelu:
            parser.error("--jump_threshold_init applies only to --model jumprelu or "
                         "--model dual --sparsity_mode jump_relu "
                         f"(got --model {args.model} --sparsity_mode {args.sparsity_mode})")
        # log(theta) is the stored parameter: theta <= 0 or non-finite has no log and would
        # either crash inside the model ctor or init every threshold to +/-inf.
        if not (math.isfinite(args.jump_threshold_init) and args.jump_threshold_init > 0):
            parser.error(f"--jump_threshold_init must be a finite value > 0, "
                         f"got {args.jump_threshold_init}")
    if not args.tied_weights and args.model != 'topk':
        parser.error(f"--no_tied_weights applies only to --model topk (got --model {args.model}); "
                     "no other trainer has a tied/untied option")
    # Same rule for --dual_train_threshold_post_warmup: only a Dual with a JumpReLU mono branch has a threshold
    # whose post-warm-up training it could change. Anywhere else it would be accepted and silently
    # ignored (train_other / train_topk have no warm-up rebuild; other Dual modes no threshold).
    if args.dual_train_threshold_post_warmup and not (args.model == 'dual'
                                                       and args.sparsity_mode == 'jump_relu'):
        parser.error("--dual_train_threshold_post_warmup applies only to --model dual "
                     "--sparsity_mode jump_relu "
                     f"(got --model {args.model} --sparsity_mode {args.sparsity_mode})")
    # Same rule for the residual-branch options: only train_dual reads them, so on any other model
    # they would be silently ignored. Combinations among them are checked by
    # train_dual.validate_ablation_options before the trainer loads any data.
    if args.model != 'dual' and (args.poly_arch != 'mlp' or args.poly_final_act != 'auto'
                                 or args.poly_warmup_zero or args.freeze_sparse_from
                                 or args.snapshot_steps):
        parser.error("--poly_arch, --poly_final_act, --poly_warmup_zero, --freeze_sparse_from and "
                     f"--snapshot_steps apply only to --model dual (got --model {args.model})")

    parsed_poly_hidden_dims = _parse_poly_hidden_dims(args.poly_hidden_dims)
    parsed_group_fractions = _parse_float_list(args.group_fractions)
    parsed_msae_nesting = _parse_int_list(args.msae_nesting)

    # Respect an explicitly provided --exp_name (needed for reproducible, resumable
    # runs and so downstream code can locate checkpoints by a known name). Only
    # auto-generate a timestamped name when none was given -- the timestamp makes
    # the path unique per run but defeats --skip_existing, so it is opt-out now.
    if not args.exp_name:
        if args.model == 'dual':
            args.exp_name = f"{args.model}_mono_coef_{args.mono_coef}_sparsity_{args.sparsity_mode}_seed_{args.seed}_time_{int(time.time())}"
        else:
            args.exp_name = f"{args.model}_mono_coef_{args.mono_coef}_seed_{args.seed}_time_{int(time.time())}"

    # Check if checkpoint already exists (early exit if --skip_existing)
    checkpoint_path = Path(args.output_dir) / args.exp_name / 'autoencoder_final.pt'
    if args.skip_existing and checkpoint_path.exists():
        print(f"Checkpoint already exists at {checkpoint_path}")
        print("Skipping training (--skip_existing is set)")
        return None

    # Built-in feature layout (see extract_imagenet.py / extract_open_images.py):
    #   <features_root>/<dataset>/<split dir>/<lmdb name>
    # ImageNet-1K has no labelled test split, so its test path is None.
    feature_files = {
        'clip': 'clip_image_features_CLIP-ViT-L-14-DataComp.lmdb',
        'vit': 'vit_features_vit-large-p16-224.lmdb',
        'sigclip': 'siglip2_image_features_siglip2-so400m-p14-384.lmdb',
        'dino': 'dino_features_dinov2-large.lmdb',
    }
    split_dirs = {
        'imagenet': {'train': 'train', 'val': 'validation', 'test': None},
        'open_images': {'train': 'train', 'val': 'valid', 'test': 'test'},
    }
    dataset_paths = {
        ds: {enc: {split: (str(Path(args.features_root) / ds / d / fname) if d else None)
                   for split, d in dirs.items()}
             for enc, fname in feature_files.items()}
        for ds, dirs in split_dirs.items()
    }

    # Only consult the built-in table when at least one path was left unspecified, and only
    # for a dataset/extractor pair the table actually covers.  A language cell passes all three
    # paths explicitly and never touches it.
    if args.train_path is None or args.val_path is None or args.test_path is None:
        try:
            selected_paths = dataset_paths[args.dataset][args.extractor]
        except KeyError:
            raise SystemExit(
                f"no built-in path table for --dataset {args.dataset} --extractor "
                f"{args.extractor}; pass --train_path/--val_path/--test_path explicitly "
                f"(this is the normal route for the language-domain cells)"
            )
        if args.train_path is None:
            args.train_path = selected_paths['train']
        if args.val_path is None:
            args.val_path = selected_paths['val']
        if args.test_path is None:
            args.test_path = selected_paths['test']

    # Resolve the per-sample standardization flag ONCE so every trainer branch below reads the
    # same value.  'auto' reproduces the historical behaviour exactly: the topk branch used to
    # hard-code normalize=True regardless of --normalize, and every other branch passed
    # --normalize through.
    if args.force_normalize == 'on':
        resolved_normalize = True
    elif args.force_normalize == 'off':
        resolved_normalize = False
    else:
        resolved_normalize = True if args.model == 'topk' else bool(args.normalize)
    args.resolved_normalize = resolved_normalize

    print(f"Dataset={args.dataset}, extractor={args.extractor}")
    print(f"Dataset paths loaded: train={args.train_path is not None}, "
          f"val={args.val_path is not None}, test={args.test_path is not None}")
    
    # Validate that train_path is provided
    if args.train_path is None:
        parser.error("train_path must be specified in dataset config file")
    
    # Import appropriate training module based on model type
    if args.model == 'topk':
        from train_topk import TrainingConfig, train_autoencoder
        config = TrainingConfig(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            exp_name=args.exp_name,
            n_latents=args.n_latents,
            activation=args.activation,
            topk_k=args.topk_k,
            tied_weights=args.tied_weights,
            normalize=resolved_normalize,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            eps=args.eps,
            auxk_coef=args.auxk_coef,
            auxk_k=args.auxk_k,
            mono_coef=args.mono_coef,
            mono_period=args.mono_period,
            dead_steps_threshold=args.dead_steps_threshold,
            dead_check_interval=args.dead_check_interval,
            wandb_project=args.wandb_project,
            wandb_name=args.exp_name,
            wandb_group=args.wandb_group,
            num_workers=args.num_workers,
            seed=args.seed,
            steps_per_epoch=args.steps_per_epoch,
            multilabel=args.multilabel,
            labels_lmdb=args.labels_lmdb,
            dataset_name=args.dataset,
            extractor_name=args.extractor,
            output_dir=args.output_dir,
            device=args.device,
            tb_dir=args.tb_dir,
            skip_extra_eval=args.skip_extra_eval
        )
    elif args.model == 'dual':
        from train_dual import TrainingConfig, train_autoencoder
        config = TrainingConfig(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            exp_name=args.exp_name,
            n_latents=args.n_latents,
            poly_hidden=args.poly_hidden,
            poly_layers=args.poly_layers,
            poly_hidden_dims=parsed_poly_hidden_dims,
            poly_arch=args.poly_arch,
            poly_final_act=args.poly_final_act,
            poly_warmup_zero=args.poly_warmup_zero,
            freeze_sparse_from=args.freeze_sparse_from,
            snapshot_steps=_parse_int_list(args.snapshot_steps),
            top_k=args.topk_k,
            sparsity_mode=args.sparsity_mode,
            # Inert unless sparsity_mode='gated' (train_dual reads it only in
            # that ctor branch), passed unconditionally like the matryoshka knobs.
            gated_sparsity_warmup=args.gated_sparsity_warmup,
            allow_degenerate_warmup=args.allow_degenerate_warmup,
            l1_coef=args.l1_coef,
            bandwidth=args.bandwidth,
            # None unless --jump_threshold_init was passed; train_dual forwards it into the
            # model_cfg only when not None and sparsity_mode == 'jump_relu'.
            jump_threshold_init=args.jump_threshold_init,
            # False unless --dual_train_threshold_post_warmup (validated above to be dual +
            # jump_relu only); a training behaviour, never forwarded into model_cfg.
            dual_train_threshold_post_warmup=args.dual_train_threshold_post_warmup,
            normalize=resolved_normalize,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            eps=args.eps,
            span_coef=args.span_coef,
            sim_coef=args.sim_coef,
            active_only_span=args.active_only_span,
            mono_coef=args.mono_coef,
            mono_period=args.mono_period,
            warmup_steps=args.warmup_steps,
            warmup_mono_target=args.warmup_mono_target,
            warmup_check_interval=args.warmup_check_interval,
            poly_lr_factor=args.poly_lr_factor,
            dead_steps_threshold=args.dead_steps_threshold,
            dead_check_interval=args.dead_check_interval,
            save_val_embeds=args.save_val_embeds,
            save_embeds_interval=args.save_embeds_interval,
            wandb_project=args.wandb_project,
            wandb_name=args.exp_name,
            wandb_group=args.wandb_group,
            num_workers=args.num_workers,
            seed=args.seed,
            steps_per_epoch=args.steps_per_epoch,
            multilabel=args.multilabel,
            labels_lmdb=args.labels_lmdb,
            dataset_name=args.dataset,
            extractor_name=args.extractor,
            output_dir=args.output_dir,
            device=args.device,
            tb_dir=args.tb_dir,
            skip_extra_eval=args.skip_extra_eval,
            skip_epoch_checkpoints=args.skip_epoch_checkpoints,
            select_from_epoch=args.select_from_epoch,
            group_fractions=parsed_group_fractions,
            msae_nesting=parsed_msae_nesting,
            msae_weighting=args.msae_weighting,
            msae_soft_cap=args.msae_soft_cap
        )
    else:  # batch_topk, vanilla, jumprelu, gated, matryoshka, msae
        from train_other import TrainingConfig, train_autoencoder
        config = TrainingConfig(
            model_type=args.model,
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            exp_name=args.exp_name,
            n_latents=args.n_latents,
            topk_k=args.topk_k,
            normalize=resolved_normalize,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            auxk_coef=args.auxk_coef,
            auxk_k=args.auxk_k,
            l1_coef=args.l1_coef,
            bandwidth=args.bandwidth,
            # None unless --jump_threshold_init was passed; train_other forwards it into the
            # JumpReLU model_cfg only when not None. Validated above to be jumprelu-only.
            jump_threshold_init=args.jump_threshold_init,
            mono_coef=args.mono_coef,
            mono_period=args.mono_period,
            dead_steps_threshold=args.dead_steps_threshold,
            wandb_project=args.wandb_project,
            wandb_name=args.exp_name,
            wandb_group=args.wandb_group,
            num_workers=args.num_workers,
            seed=args.seed,
            steps_per_epoch=args.steps_per_epoch,
            multilabel=args.multilabel,
            labels_lmdb=args.labels_lmdb,
            dataset_name=args.dataset,
            extractor_name=args.extractor,
            output_dir=args.output_dir,
            device=args.device,
            tb_dir=args.tb_dir,
            skip_extra_eval=args.skip_extra_eval,
            group_fractions=parsed_group_fractions,
            msae_nesting=parsed_msae_nesting,
            msae_weighting=args.msae_weighting,
            msae_soft_cap=args.msae_soft_cap,
            # GatedSAE knob: inert for every other model_type (train_other reads
            # it only inside the 'gated' constructor branch), passed
            # unconditionally.
            gated_sparsity_warmup=args.gated_sparsity_warmup,
        )

    # Save config to output directory: output_dir/exp_name
    output_path = Path(args.output_dir) / args.exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_save_path = output_path / 'training_config.json'
    # utils.config_payload == vars(config) minus opt-in fields left unset (e.g. jump_threshold_init
    # when --jump_threshold_init is not passed), so re-running any pre-existing cell writes the
    # same keys as its original training_config.json.  The trainers use the same helper for the
    # checkpoint 'config' entry and W&B.  Imported here, not at the top, because main.py only
    # imports torch-dependent modules after parsing (the trainers above are imported lazily too).
    from utils import config_payload
    with open(config_save_path, 'w') as f:
        json.dump(config_payload(config), f, indent=2)
    print(f"Training configuration saved to {config_save_path}")
    
    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Experiment:       {config.exp_name}")
    print(f"Output directory: {output_path}")
    print(f"\nModel:")
    if hasattr(config, 'model_type'):
        print(f"  Model type:     {config.model_type}")
    elif hasattr(config, 'activation'):
        print(f"  Model type:     topk")
    else:
        print(f"  Model type:     dual")
    print(f"  Latents:        {config.n_latents}")
    if hasattr(config, 'activation'):
        print(f"  Activation:     {config.activation}")
        if config.activation == 'topk':
            print(f"  TopK K:         {config.topk_k}")
        print(f"  Tied weights:   {config.tied_weights}")
    elif hasattr(config, 'top_k'):
        print(f"  TopK K:         {config.top_k}")
    if hasattr(config, 'poly_hidden'):
        print(f"  Poly Hidden:    {config.poly_hidden}")
        print(f"  Poly Layers:    {config.poly_layers}")
        if hasattr(config, 'poly_hidden_dims') and config.poly_hidden_dims:
            print(f"  Poly Hidden Dims: {config.poly_hidden_dims}")
        print(f"  Sparsity mode:  {config.sparsity_mode}")
        print(f"  Span Coef:      {config.span_coef}")
        print(f"  Sim Coef:       {config.sim_coef}")
        if config.warmup_steps > 0 or config.warmup_mono_target > 0:
            print(f"\n  Warm Start:")
            if config.warmup_steps > 0:
                print(f"    Steps:        {config.warmup_steps}")
            if config.warmup_mono_target > 0:
                print(f"    Mono target:  {config.warmup_mono_target:.4f}")
                print(f"    Check every:  {config.warmup_check_interval} steps")
            if config.poly_lr_factor != 1.0:
                print(f"    Poly LR factor: {config.poly_lr_factor} (post-warmup)")
            # Echoed only when set, so the log of every default run is unchanged.
            if getattr(config, 'dual_train_threshold_post_warmup', False):
                print("    JumpReLU threshold: TRAINED after warm-up (opt-in)")
    print(f"  Normalize:      {config.normalize}")
    print(f"\nData:")
    print(f"  Train:          {config.train_path}")
    if config.val_path:
        print(f"  Val:            {config.val_path}")
    if config.test_path:
        print(f"  Test:           {config.test_path}")
    print(f"\nTraining:")
    print(f"  Batch size:     {config.batch_size}")
    print(f"  Num epochs:     {config.num_epochs}")
    print(f"  Learning rate:  {config.lr}")
    if hasattr(config, 'auxk_coef'):
        print(f"  AuxK coef:      {config.auxk_coef}")
    if hasattr(config, 'l1_coef') and config.l1_coef > 0:
        print(f"  L1 coef:        {config.l1_coef}")
    if hasattr(config, 'bandwidth'):
        print(f"  Bandwidth:      {config.bandwidth}")
    print(f"  Mono coef:      {config.mono_coef}")
    if config.mono_coef > 0:
        if config.mono_period == 1:
            print(f"  Mono period:    every step (batch-level)")
        else:
            print(f"  Mono period:    every {config.mono_period} steps (full dataset)")
    print(f"\nSystem:")
    print(f"  Device:         {config.device}")
    print(f"  Num workers:    {config.num_workers}")
    print(f"  Seed:           {config.seed}")
    if config.wandb_project:
        print(f"\nLogging:")
        print(f"  Wandb project:  {config.wandb_project}")
        print(f"  Wandb name:     {config.wandb_name}")
        if config.wandb_group:
            print(f"  Wandb group:    {config.wandb_group}")
    print("="*60 + "\n")
    
    # Train the model
    autoencoder = train_autoencoder(config)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Final model saved to: {output_path / 'autoencoder_final.pt'}")
    print("="*60)
    
    return autoencoder


if __name__ == '__main__':
    main()

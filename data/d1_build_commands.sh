python data/build_finetune_corpus.py --out data/finetune_d1 \
    --min-val-per-author 1 --min-per-author 20 --max-repeat 4
python data/tokenize_shards.py --input data/finetune_d1 --out data/finetune_d1
# ship as plm-stage-d1-finetune; plan helper now sizes off the rebalanced
# train stats: ~59 epochs ≈ 1.0M tokens, still minutes of GPU

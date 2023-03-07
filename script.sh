python3 -m splade.all   config.checkpoint_dir=experiments/debug/checkpoint   config.index_dir=experiments/debug/index   config.out_dir=experiments/debug/out

slurmRun --cmd="python3 -m splade.all   config.checkpoint_dir=experiments/debug/checkpoint   config.index_dir=experiments/debug/index   config.out_dir=experiments/debug/out" --outputDir=output/newRun/

slurmRun --cmd="python -m splade.index   init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil   config.pretrained_no_yamlconfig=true   config.index_dir=experiments/pre-trained/index" --outputDir=output/Indexcocondenser/

slurmRun --cmd="python3 -m splade.all   config.checkpoint_dir=experiments/myTrain/checkpoint   config.index_dir=experiments/myTrain/index   config.out_dir=experiments/myTrain/out" --outputDir=output/myTrain/

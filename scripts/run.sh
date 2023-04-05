#!/bin/bash
#SBATCH --job-name=${experimentName}
#SBATCH --output=experiments/${experimentName}/logs/run.out
#SBATCH --error=experiments/${experimentName}/logs/run.err
#SBATCH --gres=gpu:2
#SBATCH --mem=80G 
#SBATCH --qos=normal
unset SPLADE_CONFIG_NAME
unset SPLADE_CONFIG_FULLPATH
export CUDA_VISIBLE_DEVICES=${CUDASet}
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil_24G"
python3 -m splade.all  config.checkpoint_dir=experiments/${experimentName}/checkpoint   config.index_dir=experiments/${experimentName}/index   config.out_dir=experiments/${experimentName}/out ${Lossweight}
unset SPLADE_CONFIG_NAME
export SPLADE_CONFIG_FULLPATH="/home/taoyang/research/research_everyday/projects/DR/splade/splade/experiments/${experimentName}/checkpoint/config.yaml"
for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq
do
    python3 -m splade.beir_eval \
      +beir.dataset=$$dataset \
      +beir.dataset_path=data/beir \
      config.index_retrieve_batch_size=32
done
exit
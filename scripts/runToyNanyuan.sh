#!/bin/bash
#SBATCH --job-name=${experimentName}
#SBATCH --output=experiments/${experimentName}/logs/run.out
#SBATCH --error=experiments/${experimentName}/logs/run.err
#SBATCH --time=00:25:00
#SBATCH --mem=30GB 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=titan-giant
unset SPLADE_CONFIG_NAME
unset SPLADE_CONFIG_FULLPATH
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil_psuedoWithHardToy"
python3 -m splade.all  config.checkpoint_dir=experiments/${experimentName}/checkpoint   config.index_dir=experiments/${experimentName}/index   config.out_dir=experiments/${experimentName}/out ${Lossweight}
unset SPLADE_CONFIG_NAME
export SPLADE_CONFIG_FULLPATH="/home/taoyang/research/research_everyday/projects/DR/splade/splade/experiments/${experimentName}/checkpoint/config.yaml"
# for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq
for dataset in scifact 
do
    python3 -m splade.beir_eval \
      +beir.dataset=$$dataset \
      +beir.dataset_path=data/beir \
      config.index_retrieve_batch_size=32
done
exit
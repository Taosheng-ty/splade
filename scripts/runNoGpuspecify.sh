#!/bin/bash
#SBATCH --job-name=${experimentName}
#SBATCH --output=${experimentsFolder}/${experimentName}/logs/run.out
#SBATCH --error=${experimentsFolder}/${experimentName}/logs/run.err
#SBATCH --time=2-23:00:00
#SBATCH --mem=120GB 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
unset SPLADE_CONFIG_NAME
unset SPLADE_CONFIG_FULLPATH
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil_44G"
python3 -m splade.all  config.checkpoint_dir=${experimentsFolder}/${experimentName}/checkpoint   config.index_dir=${experimentsFolder}/${experimentName}/index   config.out_dir=${experimentsFolder}/${experimentName}/out ${Lossweight}
unset SPLADE_CONFIG_NAME
export SPLADE_CONFIG_FULLPATH="/home/taoyang/research/research_everyday/projects/DR/splade/splade/${experimentsFolder}/${experimentName}/checkpoint/config.yaml"
for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq
do
    python3 -m splade.beir_eval \
      +beir.dataset=$$dataset \
      +beir.dataset_path=data/beir \
      config.index_retrieve_batch_size=128
done
exit
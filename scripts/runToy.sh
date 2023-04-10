#!/bin/bash
#SBATCH --job-name=${experimentName}
#SBATCH --output=${experimentsFolder}/${experimentName}/logs/run.out
#SBATCH --error=${experimentsFolder}/${experimentName}/logs/run.err
#SBATCH --time=1:00:00
#SBATCH --mem=10GB 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2080ti:1
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest

## SBATCH --account=soc-gpu-kp
## SBATCH --partition=soc-gpu-kp

unset SPLADE_CONFIG_NAME
unset SPLADE_CONFIG_FULLPATH
# export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil_psuedoWithHardToy"
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil_psuedoWithHardToyContrast"
python3 -m splade.all  config.checkpoint_dir=${experimentsFolder}/${experimentName}/checkpoint   config.index_dir=${experimentsFolder}/${experimentName}/index   config.out_dir=${experimentsFolder}/${experimentName}/out ${Lossweight}
unset SPLADE_CONFIG_NAME
# experimentPath="/uufs/chpc.utah.edu/common/home/u1265233/document/projects/splade/${experimentsFolder}"
# experimentPath="/home/taoyang/research/research_everyday/projects/DR/splade/splade/${experimentsFolder}"
export SPLADE_CONFIG_FULLPATH="/uufs/chpc.utah.edu/common/home/u1265233/document/projects/splade/${experimentsFolder}/${experimentName}/checkpoint/config.yaml"
# for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq
for dataset in scifact 
do
    python3 -m splade.beir_eval \
      +beir.dataset=$$dataset \
      +beir.dataset_path=data/beir \
      config.index_retrieve_batch_size=32
done
exit
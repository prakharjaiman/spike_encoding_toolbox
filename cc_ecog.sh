#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0:5:00 # DD-HH:MM:SS
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=0-999
#SBATCH --job-name=ecog_encoding

echo "Moving files"
cp -r $HOME/spike_encoding_toolbox $SLURM_TMPDIR/spike_encoding_toolbox
cd $SLURM_TMPDIR/spike_encoding_toolbox

echo "Starting application"
mkdir -p "$HOME/ecog_results/"

if $HOME/env/bin/python run_random_experiment.py --seed $SLURM_ARRAY_TASK_ID ; then
    echo "Copying results"
    mv "ecog_$SLURM_ARRAY_TASK_ID.csv" "$HOME/ecog_results/"
fi

wait
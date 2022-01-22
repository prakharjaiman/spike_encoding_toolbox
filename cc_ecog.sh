#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00 # DD-HH:MM:SS
#SBATCH --mem-per-cpu=2GB
#SBATCH --array=0-5000
#SBATCH --job-name=ecog_encoding

echo "Moving files"
cp -r $HOME/spike_encoding_toolbox $SLURM_TMPDIR/spike_encoding_toolbox
cd $SLURM_TMPDIR/spike_encoding_toolbox

echo "Starting application"
mkdir -p "$HOME/ecog_results/"

if $HOME/env/bin/python evaluate_encoder.py run_random_experiment $SLURM_ARRAY_TASK_ID; then
    echo "Copying results"
    mv "ecog_$SLURM_ARRAY_TASK_ID.txt" "$HOME/ecog_results/"
fi

wait
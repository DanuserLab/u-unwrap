#!/bin/bash
#SBATCH --job-name=unwrap_blebb
#SBATCH --output=./log/unwrap_%A_%a.out
#SBATCH --error=./log/unwrap_%A_%a.err
#SBATCH --partition=super
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --array=0-15
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shiqiu.yu2@utsouthwestern.edu

source /home2/s440708/miniconda3/bin/activate unwrap2d

BASE="/archive/bioinformatics/Danuser_lab/shared/Yu_Felix_Roshan/data/Lou_data/Shared Folder with UTSW/RhoA-Rac1 in MEF ROCK-inhibitor treated"



mapfile -t CELL_LIST < <(
    find "$BASE" -type d -path "*/P*/Cell*" -maxdepth 3
)



CELL=${CELL_LIST[$SLURM_ARRAY_TASK_ID]}

if [ -z "$CELL" ]; then
    echo "No cell for task ${SLURM_ARRAY_TASK_ID}, exiting."
    exit 0
fi


P_FULL=$(basename "$(dirname "$CELL")")
P=$(echo "$P_FULL" | sed -E 's/^(P[0-9]+).*/\1/')
CELL_NAME=$(basename "$CELL")

echo "P_FULL = $P_FULL"
echo "P = $P"
echo "Cell = $CELL"

IMG="${CELL}/vcRatio_Rac1_${P}C${CELL_NAME#Cell}.stk"
MASK="${CELL}/mask.tif"
EXTRA="${CELL}/vcRatio_RhoA_${P}C${CELL_NAME#Cell}.stk"

python run_pipeline.py \
    --img "$IMG" \
    --out "$CELL" \
    --extra "$EXTRA" \
    --rerun

conda deactivate

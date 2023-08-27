#!/bin/bash
#SBATCH --job-name=generate_act
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32          # number of cores per tasks
#SBATCH --gres=gpu:8                # number of gpus
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --error=%x-%j.out           # error file name (same to watch just one file)
#SBATCH --account=ENT212162
#SBATCH --partition=gp4d

# init environment 
export PYTHONUSERBASE=$CONDA_PREFIX
export HF_HOME=/work/twsuzrf718/hf_home

# setup distrubuted environment
echo "NODELIST="$SLURM_JOB_NODELIST
export MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
export OMP_NUM_THREADS=1
echo "master addr="$MASTER_ADDR
echo "master port="$MASTER_PORT

# generate configs for huggingface accelerator
#export NNODES=$SLURM_NNODES
#export TOTAL_GPUS=$((NNODES*8))
#srun --jobid $SLURM_JOBID bash -c './generate_config.sh $SLURM_JOBID $SLURM_NODEID $MASTER_ADDR $MASTER_PORT $NNODES $TOTAL_GPUS'

# submit slurm job
#export LAUNCHER="accelerate launch"
#export LAUNCHER="deepspeed"
export LAUNCHER="python"

#export CMD="peft_lora_clm_accelerate_ds_zero3_offload.py"
#export CMD="bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --batch_size 4 --benchmark --dtype int8"
#export CMD="../examples/pytorch/gpt/multi_gpu_gpt_example.py --ckpt_path bloom-560m/c-model/1-gpu --no_detokenize --time"
#export CMD="../examples/pytorch/gpt/bloom_lambada.py --checkpoint-path bloom-560m/c-model/1-gpu --tokenizer-path bloom-560m --dataset-path ../datasets/lambada/lambada_test.jsonl --show-progress --inference-data-type fp16 --int8_mode 0 --batch-size 1 --tensor-para-size 1"
#export CMD="../examples/pytorch/gpt/bloom_lambada.py --checkpoint-path /work/twsuzrf718/hf_home/hub/models--bigscience--bloom/snapshots/87cbfc811fd083b08a31869d303e29dc56e29ec8/c-model/8-gpu --tokenizer-path /work/twsuzrf718/hf_home/hub/models--bigscience--bloom/snapshots/87cbfc811fd083b08a31869d303e29dc56e29ec8 --inference-data-type fp16 --dataset-path ../datasets/lambada/lambada_test.jsonl --show-progress --tensor-para-size 8 --batch-size 1 --int8_mode 1"
export CMD="bloom_generate_act_scales.py"
#srun --jobid $SLURM_JOBID bash -c 'mpirun -n 2 --allow-run-as-root python ../examples/pytorch/gpt/multi_gpu_gpt_example.py --tensor_para_size=2 --pipeline_para_size=1 --ckpt_path="bloom-560m/c-model/2-gpu"'
#srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --config_file config.$SLURM_JOBID.$SLURM_NODEID.yaml $CMD'
#srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --num_gpus 8 --num_nodes 2 --hostfile hostfile $CMD'
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER $CMD'
#mpirun -n 1 --hostfile hostlist.txt python -mca plm_rsh_args "-p $MASTER_PORT" $CMD
echo "Finish $(date)"

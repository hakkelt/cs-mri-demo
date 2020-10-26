#!/bin/sh

srun -pcpu_lowpriority --cpus-per-task=2 julia --project --optimize --math-mode=fast --check-bounds=no reconstruct.jl $1

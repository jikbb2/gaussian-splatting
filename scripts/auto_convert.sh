#!/usr/bin/env python3
# -*- coding: utf-8 -*-

declare -a run_args=(
    "method_1_05_2"
)

# +
for arg in "${run_args[@]}"; do 
    python convert_mov2jpg.py "${arg}.MOV"  --out "${arg}/images" --sample-every-frames 45 --datetime
    
    zip -r "${arg}.zip" "${arg}"
done

wait
echo "All MOVs are convert to JPG with EXIF."

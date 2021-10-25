#!/bin/bash                         

python transchannel_runner.py -1
wait
python osteosarcoma.py 1 "raw" &
python osteosarcoma.py 1 "ablation" &
python osteosarcoma.py 2 "raw" &
wait
python osteosarcoma.py 2 "ablation" &
python osteosarcoma.py 3 "raw" &
python osteosarcoma.py 3 "ablation" &
wait
python figure_generator.py 
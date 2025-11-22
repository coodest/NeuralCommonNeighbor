#! /bin/bash

# prefix=docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:ncn python NeighborOverlap.py
prefix="python NeighborOverlap.py"


# for predictor in "cn1" ; do
for predictor in "incn1cn1"; do
    echo "Processing model: $predictor"



# $prefix --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 1.0 --proboffset 0.0 --alpha 0.1  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $predictor --epochs 100 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --runs 3 --dataset wikidata

# $prefix --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.1 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $predictor --epochs 100 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --runs 3 --dataset twitter

# $prefix --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $predictor --epochs 100 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --runs 3 --dataset ppi

# $prefix --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $predictor --epochs 100 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --runs 3 --dataset dblp

# $prefix --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $predictor --epochs 100 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --runs 3 --dataset blogcatalog



# python NeighborOverlap.py --xdp 0.7 --tdp 0.7 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.35  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $predictor --epochs 100 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --runs 3 --dataset wikidata1k_multiclass --multiclass

# python NeighborOverlap.py --xdp 0.7 --tdp 0.7 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.35  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $predictor --epochs 100 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --runs 3 --dataset wikidata5k_multiclass --multiclass

python NeighborOverlap.py --xdp 0.7 --tdp 0.7 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.35  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $predictor --epochs 100 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --runs 3 --dataset wikidata10k_multiclass --multiclass
# python NeighborOverlap.py --xdp 0.7 --tdp 0.7 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.35  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $predictor --epochs 100 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --runs 3 --dataset wikidata10k_multiclass 

done






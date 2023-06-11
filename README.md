# ACA2023-PerceptronBP
Advanced Computer Architecture, Spring 2023, Final Project

# Perceptron
To run perceptron branch predictor, do:
```
git clone https://github.com/jiyuntu/gem5.git
git checkout perceptron
scons build/X86/gem5.opt -j 13 # build gem5
build/X86/gem5.opt configs/example/l3se.py --cpu-type X86O3CPU --num-cpus 1 --caches --l2cache --l3cache --l1d_size "32kB" --l1i_size "32kB" --l1d_assoc 8 --l1i_assoc 8 --l2_size "512kB" --l2_assoc 8 --l3_size "16MB" --l3_assoc 16 -I 10000000 -c "../minimap2/minimap2" -o "-a ../minimap2/test/MT-human.fa ../minimap2/test/MT-orang.fa" # example command
```

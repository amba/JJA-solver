#!/usr/bin/env gnuplot

set term pdfcairo fontscale 0.5

set output 'L0.pdf'
set xlabel 'frustration'
set ylabel 'L(I=0)/2'

set grid
set key bottom right
set title 'N_x = N_y = 16, I(γ) = sin(γ)'

plot 'data_L0.dat' using 3:5 notitle

set output 'F0.pdf'
set ylabel 'free energy'
plot 'data_L0.dat' using 3:4 notitle
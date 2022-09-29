#!/usr/bin/env gnuplot

set term pdfcairo fontscale 0.5

set output 'plot.pdf'

set xlabel 'frustration'
set ylabel 'free energy'

set grid
set key bottom right
set title 'N_x = N_y = 16, I(γ) = sin(γ)'
plot 'data.dat' every :::0::0 using 4:5  w l title 'N_{max} = 100' ,\
'data.dat' every :::1::1 using 4:5  w l title 'N_{max} = 500' ,\
'data.dat' every :::2::2 using 4:5  w l title 'N_{max} = 1000' ,\
'data.dat' every :::3::3 using 4:5  w l title 'N_{max} = 5000' ,\
'data.dat' every :::4::4 using 4:5  w l title 'N_{max} = 10000' ,\

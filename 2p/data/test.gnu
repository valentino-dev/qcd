#!/usr/local/bin/gnuplot -persist
set term cairolatex 
set output "p3f_test.tex"
set logscale y
plot 'LowerBound.dat' using 1:2 pt 0

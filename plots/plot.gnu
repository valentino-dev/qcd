#!/usr/local/bin/gnuplot -persist

set term cairolatex 

glw = 2
set style line 1 lw glw lc rgb '#cc241d'
set style line 2 lw glw lc rgb '#cc241d'

set errorbars .2

set output "CorrelationFunction.tex"
filename1 = "../data/CorrelationFunction.csv"
filename2 = "../data/CorrelationFunctionPrediction.csv"
set yrange [0.006:0.02]
set xrange [29:50]
set title "Correlation function"
set xlabel '$\tau$'
set ylabel '$\langle x(0)x(\tau)\rangle$'
set xtics
set ytics
set logscale y exp(1)
plot filename1 using 1:2:3 with yerrorbars pt 0 title "simulation data", \
     filename2 using 1:2:3 with yerrorbars pt 0 title "fit curve with bootstrapping",\
     '../data/CorrelationFunction0Prediction.csv' using 1:2:3:4 with yerrorbars pt 0 title "fit curve without bootstrapping"

set output "LowerBound.tex"
set title "Lower bound"
set xrange [12:75]
unset yrange
set xtics 3
set ytics
set xlabel '$t_{\text{lower}}$'
set ylabel '$\log_e[\chi^2/dof]$'
set ylabel '$\chi^2/$dof'
#set logscale y exp(1)
unset logscale
plot "../data/LowerBound.csv" using 1:2 title '$\chi^2$' pt 8 lw 4 lc rgb '#beaed4'

unset logscale
set output "UpperBound.tex"
set title "Upper bound"
set xrange [30:80]
set xtics 2
set ytics
set xlabel '$t_{\text{upper}}$'
set ylabel '$\chi^2/$dof'
plot "../data/UpperBound.csv" using 1:2 title '$\chi^2$' pt 8 lw 4 lc rgb '#beaed4'



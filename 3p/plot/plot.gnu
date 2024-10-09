#!/usr/local/bin/gnuplot -persist
set term cairolatex 

set grid
set key top right

set logscale y

set output "p3f_mu1.tex"
set title "3 Punkt Funktionen"
set xlabel '$t_c$'
set ylabel '$C_{\mu_i=1\nu}(t_c)^{(3)}$'
#plot for [i=1:4] '../data/p3f_pd.dat' using 0:(i+0*4)*2:(i+0*4)*2+1 with yerrorbars pt 0 title '$\nu_i='.i.'$',\
plot for [i=1:4] '../data/p3f_pd.dat' using 0:i*2+1 with lines title '',\

#set output "p3f_mu2.tex"
#set title "3 Punkt Funktionen"
#set ylabel '$C_{\mu_i=2\nu}(t_c)^{(3)}$'
#plot for [i=1:4] '../data/p3f_pd.dat' using 0:(i+1*4)*2:(i+1*4)*2+1 with yerrorbars pt 0 title '$\nu_i='.i.'$',\

#set output "p3f_mu3.tex"
#set title "3 Punkt Funktionen"
#set ylabel '$C_{\mu_i=3\nu}(t_c)^{(3)}$'
#plot for [i=1:4] '../data/p3f_pd.dat' using 0:(i+2*4)*2:(i+2*4)*2+1 with yerrorbars pt 0 title '$\nu_i='.i.'$',\

#set output "p3f_mu4.tex"
#set title "3 Punkt Funktionen"
#set ylabel '$C_{\mu_i=4\nu}(t_c)^{(3)}$'
#plot for [i=1:4] '../data/p3f_pd.dat' using 0:(i+3*4)*2:(i+3*4)*2+1 with yerrorbars pt 0 title '$\nu_i='.i.'$',\

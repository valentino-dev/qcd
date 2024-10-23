set term cairolatex
set output 'p3f.tex'

set ylabel '$C_{\mu\nu}^{(3)}(t_c-t_i)$'
set xlabel '$t_c-t_i$'
mu=1
set output 'p3f_mu'.mu.'.tex'
set title 'Correlation function with $\mu='.mu.'$'
plot for [i=1:4] '../data/p3f_pd.dat' using 0:(column((i+(mu-1)*4)*2-1)):(column((i+(mu-1)*4)*2)) with yerrorbars pt 0 title '$\nu='.i.'$'

mu=2
set output 'p3f_mu'.mu.'.tex'
set title 'Correlation function with $\mu='.mu.'$'
plot for [i=1:4] '../data/p3f_pd.dat' using 0:(column((i+(mu-1)*4)*2-1)):(column((i+(mu-1)*4)*2)) with yerrorbars pt 0 title '$\nu='.i.'$'

mu=3
set output 'p3f_mu'.mu.'.tex'
set title 'Correlation function with $\mu='.mu.'$'
plot for [i=1:4] '../data/p3f_pd.dat' using 0:(column((i+(mu-1)*4)*2-1)):(column((i+(mu-1)*4)*2)) with yerrorbars pt 0 title '$\nu='.i.'$'

mu=4
set output 'p3f_mu'.mu.'.tex'
set title 'Correlation function with $\mu='.mu.'$'
plot for [i=1:4] '../data/p3f_pd.dat' using 0:(column((i+(mu-1)*4)*2-1)):(column((i+(mu-1)*4)*2)) with yerrorbars pt 0 title '$\nu='.i.'$'

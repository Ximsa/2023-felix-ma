reset
set xlabel "Dataset label"
set ylabel "Amount of vertices"
set style fill solid border lt -1
set boxwidth 0.8
set xrange [-0.5:39.5]
set grid
set key top right
plot for [i=1:5] 'cora_transposed.dat' using 6-i title sprintf("Step %i",6-i) linetype i+1 with boxes, "cora_transposed.dat" using 6 with points lt 1 ps 1.6 title "Dataset"

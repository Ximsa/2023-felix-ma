names="disabled 0 0.05 0.1 0.2 0.4 0.8 1"
set key right bottom
set datafile separator ";"

plot for [J=0:7] 'label_prop_threshold_influcence_cora.csv' every ::3+J*21::23+J*21 using 3:10 with lines title word(names, J+1)
pause -1
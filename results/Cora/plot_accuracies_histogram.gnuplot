reset


FILE=ARG1
FILE="GPN-GCN-own-medoids.csv"
MEASURE="Test class accuracies"
COL_NUM=system("head -n 1 ".FILE." | tr \";\" \"\n\" | grep -n \"".MEASURE."\" | cut -d \":\" -f 1")
load '< echo "\$DATA << EOD" & cut -d ";" -f '.COL_NUM.' '.FILE.' | head -n 7 | tail -n 6 | tr -d "[]" | csvtool transpose - | tr "," " "'
stats $DATA
print $DATA
set xlabel "Dataset label"
set ylabel "Class recall"

set style data histogram
set style fill solid border lt -1
set boxwidth 0.8
set yrange [0:1]
set xrange [-0.5:STATS_columns+0.5]
set grid
set key top right
plot for [i=1:STATS_columns+1] $DATA using i title sprintf("Step %i",i) linetype i+1 
pause -1
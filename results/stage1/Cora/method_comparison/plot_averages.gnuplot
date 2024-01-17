# plot directory with runs
set datafile separator ";"
MEASURE = 'Unlabeled macro-f1'
if (exists("ARG1")){MEASURE=ARG1}
X_SCALE = 'Budget used'

FILES = system("ls -1 *.csv")
TOTAL_ROWS = 0
do for [FILE in FILES]{
    load '< echo "\$DATA << EOD" & cat '.FILE
    stats $DATA using X_SCALE nooutput
    TOTAL_ROWS = TOTAL_ROWS + STATS_records/STATS_blocks}
array BUDGET[TOTAL_ROWS]
array MEAN[TOTAL_ROWS]
array STDDEV[TOTAL_ROWS]
array LEGEND[TOTAL_ROWS]
i = 0
do for [FILE in FILES]{
    unset xrange # important for stats to work
    unset yrange
    load '< echo "\$DATA << EOD" & cat '.FILE
    stats $DATA using X_SCALE:MEASURE nooutput
    ROWCOUNT = STATS_records/STATS_blocks
    LEGEND[i/ROWCOUNT+1] = FILE
    array LOC_MEAN[ROWCOUNT]
    array LOC_STDDEV[ROWCOUNT]
    do for [j=1:ROWCOUNT] {
	stats $DATA every ::j-1::j-1 using X_SCALE:MEASURE nooutput
	BUDGET[i+j] = STATS_mean_x
	MEAN[i+j] = STATS_mean_y
	LOC_MEAN[j] = STATS_mean_y
	STDDEV[i+j] = STATS_stddev_y
	LOC_STDDEV[j] = STATS_stddev_y}
    i = i + ROWCOUNT
    print sprintf("%s\tMean_Std(after 5|C|): %.1f_{%.1f}\t %.1f_{%.1f}", FILE, LOC_MEAN[ROWCOUNT]*100, LOC_STDDEV[ROWCOUNT]*100, LOC_MEAN[6]*100, LOC_STDDEV[6]*100)}
set grid
set yrange [0:1]
set xtics BUDGET[3] # every second datapoint
set xlabel X_SCALE
set ytics
set ylabel MEASURE
set key bottom
plot for [i=0:TOTAL_ROWS/ROWCOUNT-1] BUDGET every ::i*ROWCOUNT::i*ROWCOUNT+ROWCOUNT-1 using (BUDGET[$1]):(MEAN[$1]) title LEGEND[i+1] with lines
pause -1
reset

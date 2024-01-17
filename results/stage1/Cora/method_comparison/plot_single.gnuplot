# plot single run
set datafile separator ";"
FILE=ARG1
load '< echo "\$DATA << EOD" & cat '.FILE
MEASURE = 'unlabeled accuracy'
if (exists("ARG2")){MEASURE=ARG2}
X_SCALE = 'budget used'
set ylabel MEASURE
set xlabel X_SCALE

stats $DATA using X_SCALE:MEASURE nooutput
ROWCOUNT = STATS_records/STATS_blocks

unset xrange # important for stats to work
unset yrange

# fill candlestick plot data
array BUDGET[ROWCOUNT]
array LOW[ROWCOUNT]
array LOW_QUART[ROWCOUNT]
array MEAN[ROWCOUNT]
array HIGH_QUART[ROWCOUNT]
array HIGH[ROWCOUNT]

do for [i=1:ROWCOUNT] {
    stats $DATA every ::i-1::i-1 using X_SCALE:MEASURE nooutput
    BUDGET[i] = STATS_mean_x
    LOW[i] = STATS_min_y
    LOW_QUART[i] = STATS_lo_quartile_y
    MEAN[i] = STATS_mean_y
    HIGH_QUART[i] = STATS_up_quartile_y
    HIGH[i] = STATS_max_y
}
set grid
set xtics BUDGET[3] # every second datapoint
set yrange [0:1]
set xrange [-BUDGET[2]/2:BUDGET[ROWCOUNT]+BUDGET[2]/2]
plot BUDGET using (BUDGET[$1]):(LOW_QUART[$1]):(LOW[$1]):(HIGH[$1]):(HIGH_QUART[$1]) with candlesticks title "Quartiles" whiskerbars ,\
     '' using (BUDGET[$1]):(MEAN[$1]):(MEAN[$1]):(MEAN[$1]):(MEAN[$1]) with candlesticks notitle,\
     '' using (BUDGET[$1]):(MEAN[$1]) with lines title sprintf("Average %s",MEASURE),\
     '' using (BUDGET[$1]):(LOW_QUART[$1]) with lines notitle lt 0,\
     '' using (BUDGET[$1]):(HIGH_QUART[$1]) with lines notitle lt 0
pause -1
reset
     
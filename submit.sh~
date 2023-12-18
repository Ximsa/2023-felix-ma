#/bin/sh
FILES=$(ls -1 config/*.edn)
for FILE in $FILES
do
    echo $FILE
    CONFIG=$FILE sbatch -p gpu_8 job_omp.sh
done

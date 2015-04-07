#!/bin/bash


CNT=3

for (( i=0; i<6000; i++ ))
do

    # STD: 1111.3330 NEGS: 2222
    RESULT=`/c/Working/python_portable/App/python train.py | grep STD`
    STD=`echo $RESULT | awk '{print $2}'`
    NEGS=`echo $RESULT | awk '{print $4}'`

    STD_THR=`echo "$STD 2500000" | awk '{print ($1 < $2)}'`
    STD_THR2=`echo "$STD 1000000" | awk '{print ($1 > $2)}'`

    if [ $STD_THR -eq 1 ]
    then
        if [ $STD_THR2 -eq 1 ]
        then
            if [ $NEGS -eq 0 ]
            then
                mv ../../submission_t.txt "../../submission_at_$CNT.txt"
                CNT=$(( $CNT + 1 ))
                echo $STD"; "$NEGS
            else
                echo "NEGS: "$NEGS
            fi
        else
            echo "STD2: "$STD
        fi
    else
        echo "STD: "$STD
    fi

    echo "iter: "$i
done


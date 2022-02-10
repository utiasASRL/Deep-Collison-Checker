#!/bin/bash

#####################
# Common parameters #
#####################

for i in {1..5}
do

  for MAPPING in "2"
  do
    for ARGS in "-fg"
    do
      # for PARAMS in "RandWand_params"
      for PARAMS in "RandFlow_params"
      do
          for TOUR in "A_tour" "B_tour" "C_tour"
          #for TOUR in "A_tour"
          do

              echo ""
              echo "|--------------------------------------------------------------------------"
              echo "|"
              echo "|    Starting new experiment with $PARAMS on $TOUR"
              echo "|    *************************************************"
              echo "|"
              echo "|"
              echo ""
              
              sleep 5

              ./master.sh $ARGS -m $MAPPING -t $TOUR -p $PARAMS
                
              sleep 5
          done
      done
    done
  done

done

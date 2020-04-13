#!/bin/sh
i=0
for MODEL in 'simple' 'full'
do
	for ACTIVATION in 'elu' 'identity'
	do
		if [ $ACTIVATION = 'elu' ];then
       			ACTIVATION_NAME=''
		else
       			ACTIVATION_NAME='_linear'
		fi
		for ORDER in 2 3 4
		do
        		if [ $ORDER = 2 ]; then
                		ORDER_NAME="2nd"
        		elif [ $ORDER = 3 ]; then
                		ORDER_NAME="3rd"
        		elif [ $ORDER = 4 ]; then
               			ORDER_NAME="4th"
        		fi
			for NORMALIZE in "2nd" "final"
			do
				for RESIDUAL in "False" "True"
				do
					if [ $RESIDUAL = "True" ]; then
        					RESIDUAL_NAME="_residual"
					else
        					RESIDUAL_NAME=""
					fi
					if ! ( [ $ORDER = 2 ] && [ $NORMALIZE = "2nd" ] ); then
						if ! ( ( [ $MODEL = "simple" ] || [ $ORDER = 2 ] ) && [ $RESIDUAL = "True" ] ); then
							echo "TPAMI/${ORDER_NAME}_order_${MODEL}_norm_${NORMALIZE}${RESIDUAL_NAME}${ACTIVATION_NAME}"
							echo $(python spiral_higher_order.py --name sliced --dataset COMA --mode test --order ${ORDER} --model ${MODEL}  --normalize ${NORMALIZE} --residual ${RESIDUAL} --activation ${ACTIVATION}  --results_folder TPAMI/${ORDER_NAME}_order_${MODEL}_norm_${NORMALIZE}${RESIDUAL_NAME}${ACTIVATION_NAME} --device_idx 8  --batch_size 16)
							i=$((i+1))
						fi
					fi
				done
			done
		done
	done
done

echo $i
#echo "TPAMI/${ORDER_NAME}_order_${MODEL}_norm_${NORMALIZE}${RESIDUAL_NAME}${ACTIVATION_NAME}"
#python spiral_higher_order.py --name sliced --dataset COMA --mode test --order ${order} --model ${model}  --normalize ${normalize} --residual ${residual} --activation ${activation}  --results_folder TPAMI/${order_name}_order_${model}_norm_${normalize}${residual_name}${activation_name} --device_idx 5 --batch_size 16



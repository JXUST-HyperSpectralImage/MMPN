for patch_bands in '10' '20' '40' '50' '100' '200'
do
for batch_size in '8' '16' '24' '32' '40'
do
for patch_size in '9' '11' '13' '15' '17'
do
for ((i = 0; i < 10; i++ ))
do
# echo $patch_bands $batch_size $patch_size
python main.py --dataset IndianPines --load_data 0.10 --validation_percentage 0.10 --patch_bands $patch_bands --batch_size $batch_size --lr 0.01 --class_balancing --patch_size $patch_size --epoch 120
done
done
done
done

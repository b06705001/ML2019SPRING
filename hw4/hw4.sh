testing_data=$1
output_file=$2

wget 'https://github.com/b06705001/ML2019SPRING/releases/download/HW3-model/my_model.h5'

python HW4_1.py $testing_data $output_file
python HW4_2.py $testing_data $output_file
python HW4_2_1.py $testing_data $output_file

python HW4_3.py $testing_data $output_file

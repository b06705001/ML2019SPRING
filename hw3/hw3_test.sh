testing_data=$1
output_file=$2

wget 'https://github.com/b06705001/ML2019SPRING/releases/download/HW3-model/my_model.h5'
python HW3_test.py $testing_data $output_file
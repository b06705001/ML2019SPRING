testing_data=$1
jieba_data=$2
output_file=$3
wget "https://www.dropbox.com/s/pfnh8ky3x9h4lxs/word.model.wv.vectors.npy?dl=1" -O "word.model.wv.vectors.npy"
wget "https://www.dropbox.com/s/lzxi8425wu6wbmb/word.model.trainables.syn1neg.npy?dl=1" -O "word.model.trainables.syn1neg.npy"
wget "https://www.dropbox.com/s/eauzj290vqksf07/my_model.h5?dl=1" -O "my_model.h5"
python RNN_test.py $1 $2 $3

## install Python 3.7.10
```bash
wget http://www.python.org/ftp/python/3.7.10/Python-3.7.10.tgz
tar -zxvf Python-3.7.10.tgz
cd Python-3.7.10
mkdir ~/.localpython
./configure --prefix=/home/asmo/.localpython
make
make install
```

## get data
```bash
!wget https://www.dropbox.com/s/ppyp3izcb1us8hb/training_sample.csv?dl=1
!mv training_sample.csv?dl=1 Data/training_sample.csv

!wget https://www.dropbox.com/s/c4n74myrw5ryd02/dataset_test.csv?dl=1
!mv dataset_test.csv?dl=1 Data/dataset_test.csv

!wget https://www.dropbox.com/s/rhy90y1t75cst2c/full_training_data.csv?dl=1
!mv full_training_data.csv?dl=1 Data/full_training_data.csv

!wget https://www.dropbox.com/s/ppyp3izcb1us8hb/training_sample.csv?dl=1
!mv training_sample.csv?dl=1 Data/training_sample.csv
```

## dependencies
```bash
pip install -r requirements.txt
pip install 'huggingface_hub[cli,torch]'
```

## set huggingface token
```bash
huggingface-cli login
```

## run job

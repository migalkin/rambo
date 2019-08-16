git clone https://github.com/geraltofrivia/mytorch.git
cd mytorch
chmod +x setup.sh
./setup.sh


mkdir data/parsed_data
mkdir data/pre_training_data

cd data/raw_data
tar -xvf fb15k_wd.tar.gz

cd ../..
git clone https://github.com/geraltofrivia/mytorch.git
cd mytorch
chmod +x setup.sh
./setup.sh
cd ..

mkdir data/parsed_data
mkdir data/pre_training_data

cd data/raw_data/fb15k_wd
tar -xvf fb15k_wd.tar.gz

cd ../../..

# Cloning OpenKE (for their data)
git clone https://github.com/thunlp/OpenKE.git

# for dataset - WikiPeople
# git clone ___
# run script.

echo "We're done pulling data. Now lets run a data parser to be on our toes."
python parse_wd15k.py
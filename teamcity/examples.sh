echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo 'executing ./bin/_check_pipeline.sh'
bash ./bin/tests/_check_pipeline.sh

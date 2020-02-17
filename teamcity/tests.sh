echo "pip install -r requirements/requirements.txt"
pip install -r requirements/requirements.txt

echo "pip install -r requirements/requirements-dev.txt"
pip install -r requirements/requirements-dev.txt

echo "make check-style"
make check-style

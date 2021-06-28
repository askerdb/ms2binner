VERSION=$1

sed -i -E "/version=[^ ]/s/\".*\"/\"$VERSION\"/" setup.py
sed -i -E "/version=[^ ]/s/\".*\"/\"$VERSION\"/" meta.yaml
sed -i -E "/release = [^ ]/s/\".*\"/\"$VERSION\"/" docs/conf.py

rm -rf dist/*
python3 setup.py sdist bdist_wheel
twine upload dist/*

conda build . -c bioconda

pip3 install .

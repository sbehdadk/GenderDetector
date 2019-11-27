
#!/bin/bash
mkdir -p data
cd data

if [ ! -f imdb_crop.tar ]; then
    wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
fi

if [ ! -d imdb_crop ]; then
    tar xf imdb_crop.tar
fi

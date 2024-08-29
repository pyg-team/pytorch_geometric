#!/bin/sh

# Check if gzip is installed
if ! command -v gzip &> /dev/null
then
    echo "gzip could not be found. Please install gzip and try again."
    exit
fi

# Check if wget is installed
if ! command -v wget &> /dev/null
then
    echo "wget could not be found. Please install wget and try again."
    exit
fi

# Check if unzip is installed
if ! command -v unzip &> /dev/null
then
    echo "unzip could not be found. Please install unzip and try again."
    exit
fi

# Wikidata5m
wget -O "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz"
tar -xvf "wikidata5m_alias.tar.gz"
wget -O "https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_all_triplet.txt.gz"
gzip -d "wikidata5m_all_triplet.txt.gz" -f

# 2Multihopqa
wget -O "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip"
unzip -o "data_ids_april7.zip"
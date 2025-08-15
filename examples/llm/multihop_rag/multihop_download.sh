#!/bin/sh

# Wikidata5m

wget -O "wikidata5m_alias.tar.gz" "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz"
tar -xvf "wikidata5m_alias.tar.gz"
wget -O "wikidata5m_all_triplet.txt.gz" "https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_all_triplet.txt.gz"
gzip -d "wikidata5m_all_triplet.txt.gz" -f

# 2Multihopqa
wget -O "data_ids_april7.zip" "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip"
unzip -o "data_ids_april7.zip"

#!/bin/bash

#wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2023a_SSH_mapping_OSE/maps/NeurOST_SSH-SST_allsat-alg.tar.gz

podaac-data-downloader -c NEUROST_SSH-SST_L4_V2024.0 -d /Odyssey/public/NeurOST/2010-2020 --start-date 2020-01-01T00:00:00Z --end-date 2020-02-01T00:00:00Z -e ""
#!/usr/bin/env bash
# Extract selected stations (by aes id) from the CWFIS decade files into data/raw/stations.csv
set -euo pipefail
cd "$(dirname "$0")/.."
IDS="${IDS:-2204100,1123939,3053536}"   # Yellowknife A, Kelowna, Jasper Warden
out=data/raw/stations.csv
echo "rep_date,aes,wmo,name,lon,lat,elev,temp,td,rh,ws,wdir,precip,sog,ffmc,dmc,dc,isi,bui,fwi,dsr,opts,calcstatus" > "$out"
for f in data/raw/cwfis_fwi20{00,10,20}sv3.0_ll.csv.bz2; do
  bzcat "$f" | awk -F, -v ids="$IDS" 'BEGIN{n=split(ids,a,",");for(i=1;i<=n;i++)k[a[i]]=1} NR>1 && ($2 in k)' >> "$out"
done
wc -l "$out"

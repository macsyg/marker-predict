import json

NEW_MARKERS = ['panCK', 'HistoneH3',
       'SMA', 'CD7', 'CD11b', 'Arg1', 'CD146', 'EGFR', 'CD45', 'CD31', 'MMP9',
       'CD20', 'CD204', 'p53', 'CD3', 'Lamp3', 'CD11c', 'PD1', 'CD73', 'Bcl2',
       'GATA3', 'CD155', 'CD10', 'NKG2A', 'FOXP3', 'CXCL13', 'CD8a', 'EOMES',
       'CD137', 'CD134', 'CD209', 'CD56', 'Tbet', 'GITR', 'Ecad', 'Tim3',
       'CXCL8', 'CD66b', 'DNA1', 'DNA2', 'Ki67', 'Podoplanin', 'IgG', 'CD15']


f = open("markers.json", "r") 
d = json.load(f)
f.close()

for elem in NEW_MARKERS: 
    if elem not in d:
        d[elem] = len(d)

f = open("markers.json", "w") 
json.dump(d, f, indent='\t')
f.close()

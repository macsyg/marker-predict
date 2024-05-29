import pandas as pd

CELL_DF_PATH = '../eb_panel_2_esb_train_nsclc2_df_bin.df'

cell_df = pd.read_csv(CELL_DF_PATH)

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)


# PANEL_1_MARKER_NAMES = ['MPO', 'HistoneH3', 'SMA', 'CD16', 'CD38',
#        'HLADR', 'CD27', 'CD15', 'CD45RA', 'CD163', 'B2M', 'CD20', 'CD68',
#        'Ido1', 'CD3', 'LAG3', 'CD11c', 'PD1', 'PDGFRb', 'CD7', 'GrzB',
#        'PDL1', 'TCF7', 'CD45RO', 'FOXP3', 'ICOS', 'CD8a', 'CarbonicAnhydrase',
#        'CD33', 'Ki67', 'VISTA', 'CD40', 'CD4', 'CD14', 'Ecad', 'CD303',
#        'CD206', 'cleavedPARP', 'DNA1', 'DNA2']

PANEL_2_MARKER_NAMES = ['panCK', 'HistoneH3',
       'SMA', 'CD7', 'CD11b', 'Arg1', 'CD146', 'EGFR', 'CD45', 'CD31', 'MMP9',
       'CD20', 'CD204', 'p53', 'CD3', 'Lamp3', 'CD11c', 'PD1', 'CD73', 'Bcl2',
       'GATA3', 'CD155', 'CD10', 'NKG2A', 'FOXP3', 'CXCL13', 'CD8a', 'EOMES',
       'CD137', 'CD134', 'CD209', 'CD56', 'Tbet', 'GITR', 'Ecad', 'Tim3',
       'CXCL8', 'CD66b', 'DNA1', 'DNA2', 'Ki67', 'Podoplanin', 'IgG', 'CD15']


print(cell_df['P3'].head())
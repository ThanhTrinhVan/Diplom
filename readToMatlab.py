import pandas as pd

data = pd.read_excel('I:/GradEs/Testing/NormGoodBMP/Matrix_data.xlsx', sheet_name='non_test', sep='delimiter', header=None)
data = data.values
name = ['XRcon=', 'XRcor=', 'XRen=', 'XRhom=', 'XRentr=', 'XRidm=', 'XGcon=', 'XGcor=', 'XGen=', 'XGhom=', 'XGentr=', 'XGidm=', 'XBcon=', 'XBcor=', 'XBen=', 'XBhom=', 'XBentr=', 'XBidm=',
    'XRGcon=', 'XRGcor=', 'XRGen=', 'XRGhom=', 'XRGentr=', 'XRGidm=', 'XRBcon=', 'XRBcor=', 'XRBen=', 'XRBhom=', 'XRBentr=', 'XRBidm=', 'XGBcon=', 'XGBcor=', 'XGBen=', 'XGBhom=', 'XGBentr=', 'XGBidm=']
'''
name = ['Rcon=', 'Rcor=', 'Ren=', 'Rhom=', 'Rentr=', 'Ridm=', 'Gcon=', 'Gcor=', 'Gen=', 'Ghom=', 'Gentr=', 'Gidm=', 'Bcon=', 'Bcor=', 'Ben=', 'Bhom=', 'Bentr=', 'Bidm=',
    'RGcon=', 'RGcor=', 'RGen=', 'RGhom=', 'RGentr=', 'RGidm=', 'RBcon=', 'RBcor=', 'RBen=', 'RBhom=', 'RBentr=', 'RBidm=', 'GBcon=', 'GBcor=', 'GBen=', 'GBhom=', 'GBentr=', 'GBidm=']
'''
for i in range(0,36):
    print("{0} [{1:.7f} {2:.7f} {3:.7f} {4:.7f} {5:.7f} {6:.7f} {7:.7f} {8:.7f}];".format(name[i],
        data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6], data[i,7]))
    #print(name[i],data[i,0:9],';')
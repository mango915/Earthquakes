file = '/home/mango/Documents/unipd/Earthquakes/SouthCalifornia-1982-2011_Physics-of-Data.dat'

data = np.genfromtxt(file, skip_header=0, skip_footer=1, names=True, dtype=None, delimiter=' ')

header = ["id","id_prv","time","magnitude","x","y", "z"]



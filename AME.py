import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage


def get_npz(nb_max,pas):
#Cette fonction renvoie une liste contenant les noms des fichiers à l'extension "npz"
#nb_max (int) correspond à la valeur max des parties numériques incluent dans les noms
#pas (int) correspond à l'ecart entre les parties numériques de deux noms successifs

    i = 1
    name = []
    while i <= nb_max:
        name.append(i)
        i += pas

    init = "C 1 _ 0 0 0 0 0 0 . n p z" #cette ligne nécessite éventuellement une modification en fonction du cas: "C1" ou "C2"
    init = init.split()
    npz = []

    for i in name:
        z = 8
        i = str(i)
        i.split()
        for nb in range(1,len(i)+1):
            init[z] = i[-nb]
            z -= 1
        npz.append("".join(init))
    return npz

npz=get_npz(121,10)


def get_E2(DATA,r):
#Cette fonction renvoie la valeur de la distribution cumulative (ramenée à l'aire) pour une échelle donnée.
#Les étapes: identifier (sur l'image) la zone concernée; générer une image binaire mettant en valeur ladite zone; mesurer la surface en nb de pixels et la convertir en m^2 
#DATA (numpy.lib.npyio.NpzFile) représente la librairie contenant les données à exploiter.
#r (int/float) représente l'échelle

    fig,axes = plt.subplots(1,1)
    if r < 0:
        axes.contourf(DATA["y"], DATA["x"], DATA["levelset"], levels = [r,0])
    else:
        axes.contourf(DATA["y"], DATA["x"], DATA["levelset"], levels = [0,r])

    extent = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('test.png', bbox_inches = extent)
 
    im = np.array(Image.open('test.png').convert('L'))
    threshold = skimage.filters.threshold_otsu(im)
    im_bin = 1 * (im == threshold)
    im_bin = np.where(im_bin == 1,0,1)
    plt.imshow(im_bin,cmap="gray")
    
    surface_pixels = im.shape[0] * im.shape[1]
    surface_metres = max(DATA["x"]) * max(DATA["y"])
    dS = surface_metres / surface_pixels

    area = (len(np.where(im_bin == 0)[0])) * dS
    plt.show()
    
    return area


def get_e2(DATA,r):
#Cette fonction renvoie la valeur de la distribution d'échelles(ramenée à la longueur de l'iso-contour) pour une échelle donnée.
#les étapes: identifier l'iso-contour en discrétisant des points; faire un cumul entre les distances euclidiennes séparants deux points consécutifs
#DATA (numpy.lib.npyio.NpzFile) représente la librairie contenant les données à exploiter.
#r (int/float) représente l'échelle

    cs = plt.contour(DATA["y"],DATA["x"],DATA["levelset"],levels=[r])
    p = cs.collections[0].get_paths()
    length = 0.0
    for path in range(len(p)):
        v = p[path].vertices
        x = v[:,0]
        y = v[:,1]
        plt.scatter(x,y)
        for i in range(len(x)-1):
            length  += np.sqrt((x[i]-x[i+1])**2+(y[i]-y[i+1])**2)
    plt.show()
    
    return length


E_2=[]
e_2=[]
times=[]
scales=[]

for name in npz:
    DATA=np.load(name)
    cs = plt.contour(DATA["y"],DATA["x"],DATA["levelset"],levels=[0.0])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]
    times.append(DATA["t"])
    plt.show()
    
    r = np.linspace(np.min(DATA["y"])-np.min(x),np.max(DATA["y"])-np.max(x),len(DATA["y"]))
    # r est un tableau 1D contenant les valeurs d'échelles disponibles dans le domaine (sur l'image)              
    scales.append(r)
    
    area = np.linspace(0,1,len(r))
    length = np.linspace(0,1,len(r))
    nb = 0
    for elt in r:
        area[nb] = get_area(DATA,elt)
        length[nb] = get_length(DATA,elt)
        nb += 1
    E_2.append(area)
    e_2.append(length)



plt.figure()
for i in range(len(npz)):
    inf_ = np.where(scales[i] < 0)[0]
    sup_ = np.where(scales[i] >= 0)[0]
    
    p = plt.plot(-scales[i][inf_],np.array(E_2)[i][inf_],':')
    color = p[0].get_color()
    plt.plot(scales[i][sup_],np.array(E_2)[i][sup_],'-',label="t="+str(times[i])[:4],c=color)
plt.xlabel("|r| [m]")
plt.ylabel("$E_2$ [$m^2$] ")
plt.grid()
plt.show()


plt.figure()
for i in range(len(npz)):
    inf_ = np.where(scales[i] < 0)[0]
    sup_ = np.where(scales[i] >= 0)[0]
    
    p = plt.plot(-scales[i][inf_],e_2[i][inf_],':')
    color = p[0].get_color()
    plt.plot(scales[i][sup_],e_2[i][sup_],'-',label="t="+str(times[i])[:4],c=color)
plt.grid()
plt.xlabel("|r| [m] ")
plt.ylabel(r"$e_2$ [m] ")
plt.legend()
plt.show()

#Détermination d'échelles caractéristiques
scales_light = [] #liste qui contiendra les échelles caractéristiques du fluide le moins dense (fluide bleu)
scales_heavy = [] #liste qui contiendra les échelles caractéristiques du fluide le plus dense (fluide rouge)

for t in range(len(npz)):
    y_ = e_2[t]
    x_ = scales[t]
    derivate = []
    
    for i in range(len(x)-1):
        delta = (y_[i+1]-y_[i-1])/(x_[i+1]-x_[i-1]) # méthode des dérivées centrées
        derivate.append(delta)
        
    plt.plot(x_[1:-1],derivate[1:],"-o")
    plt.grid()
    plt.show()
    
    scale_blue = [x_[:-1][i] for i,elt in enumerate(derivate) if elt==np.max(derivate)][0]
    scale_red = [x_[:-1][i] for i,elt in enumerate(derivate) if elt==np.min(derivate)][0]
    print(scale_blue,scale_red)
    
    scales_light.append(scale_blue)
    scales_heavy.append(scale_red)
    
plt.figure()
plt.plot( times ,-np.array(scales_light), "b-o", label=r"$r_-$")
plt.plot( times, scales_heavy, "r-o", label=r"$r_+$")
plt.xlabel("t  [s]")
plt.ylabel("r  [m]")
plt.legend()
plt.grid()
plt.show()
    
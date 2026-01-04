import numpy as np
import matplotlib.pyplot as plt
import os


#oppgave2a
def _replicate_pad(img: np.ndarray, ph: int, pw: int):
    #utvider bildet ved å kopiere de nærmeste pikselverdiene
    #ph: oppfyller over og under
    #pw: oppfyller til venstre og høyre
    #Bildet fra oppgaven er et gråtonebilde, så vi bare vurdere dette tilfellet
    if img.ndim == 2:
        #Hvis det er gråtonebilde
        return np.pad(img,((ph,ph),(pw,pw)), mode = "edge")
    else:
        raise ValueError("Det er ikke et graatonebilde")
    
def konvolusjon(image: np.ndarray, kernel: np.ndarray):
    img = image.astype(np.float32, copy = False) #konverter til float32 for å unngå overflow
    kh,kw = kernel.shape       #høyde og bredde på kjernen
    H,W = img.shape           # høyden og bredde på bildet
    ph,pw = kh//2, kw//2      # Hvor mange piksler som skal paddes

    pad = _replicate_pad(img,ph,pw) # Utvid bildet ved kantene
    out = np.zeros_like(img,dtype = np.float32) #lager en tom utdata-bilde med samme dimensjoner som inndata-bildet

    for i in range(H):
        for j in range(W):
            region = pad[i:i+kh,j:j+kw]   # ekstraher aktuelt konvolusjonsområde
            out[i,j] = np.sum(region*kernel) #beregn konvolusjonsresultat

    
    return out




#oppgave2b
def gauss_kjerne(sigma: float):
    # vi skal lage en 2D- gauss kjerne, størrelsen er k = 1+ 2*ceil(4*sigma)
    k = int(1 + 2*np.ceil(4*sigma))
    ax = np.arange(-(k//2),k//2+1) #lag koordinatakser
    xx,yy = np.meshgrid(ax, ax, indexing = "xy")  #lag 2d-nett for x og y koordinater
    kjerne = np.exp(-(xx**2 + yy**2)/(2*sigma*sigma))
    kjerne = kjerne / kjerne.sum()   # normaliser til sum =1
    return kjerne.astype(np.float32)

def gradient_symmetric(img_blur: np.ndarray):   
    # Input: img_blur - bilde som allerede er glattet med Gauss-filter
    # Dette reduserer støy før kantdeteksjon
    
    # Differansekjerne for sentraldifferanse
    # [-0.5, 0.0, 0.5] beregner derivert ved å sammenligne venstre og høyre nab
    # 1D symmerisk differansekjerne for gradientberegning
     dx = np.array([[-0.5,0.0,0.5]],dtype=np.float32)
     dy = dx.T #Transponer for vertikal gradient

    # beregn gradienter i x- og y-retning
     gx = konvolusjon(img_blur,dx)  # horisontal gradient
     gy = konvolusjon(img_blur,dy)  # vertikal gradient

     #gradientstyrke og retning 
     mag = np.sqrt(gx*gx + gy*gy)    #styrke
     ang = np.arctan2(gy,gx)        # retning i radianer
     return gx, gy, mag, ang

def nms_fire_retninger(mag: np.ndarray, ang : np.ndarray):
    #Hent bildedimensjoner og opprett utdatabilde
    H,W = mag.shape
    ut = np.zeros_like(mag, dtype=np.float32)

    #Konverter vinkler fra radianer til grader og normaliser til [0,180), som kan gjøre det lettere å kategorisere kantretninger
    vinkel = np.rad2deg(ang) %180 # ang er fra forrige metode


    # Gå gjennom alle pikslene (unntatt kantpikslene)
    for rad in range(1,H-1):
        for kol in range(1,W-1):
            a = vinkel[rad,kol] # hent vinkelen for denne pikselen
            m = mag[rad,kol]   # hent gradientstyrken for denne pikselen
        
        #kategorisere vinkelen i 0/45/90/135
            if(0<= a <22.5) or (157.5<= a <180):
            #horisontal kant: sammenlign med venstre og høyre nabo
               n1= mag[rad, kol-1]
               n2= mag[rad, kol+1]
            elif 22.5<= a < 67.5:
            #45 diagonal kant : sammenlign med diagonale naboer
               n1= mag[rad-1, kol+1]
               n2= mag[rad+1, kol-1]
            elif 67.5 <=a <112.5:
            #vertikal kant: sammenlign med over og under naboer
               n1= mag[rad-1, kol]
               n2= mag[rad+1, kol]
            else: # 112.5<=a < 157.5
               n1= mag[rad-1, kol-1]
               n2= mag[rad+1, kol+1]
        
        #utfør non-maximum suppression
            if m >= n1 and m >=n2:
            # denne pikselen er et lokalt maksimum i kantretningen, da skal vi beholde den
               ut[rad, kol] = m
            #ellers pikselen er ikke et lokalt maksimum, skal undertrykke

    return ut

def hysteresis(mag_nms: np.ndarray, Th: float, Tl:float): # Th: high threshold ; Tl:Low threshold
    H,W = mag_nms.shape
    #Definerer sterke og svake kanter basert på tersklene
    strong = (mag_nms >= Th) # sterke kanter
    weak = (mag_nms >=Tl)& ~strong  # svake kanter men ikke inkluderer strong

    out = np.zeros((H,W),dtype=np.uint8)  #utbilde og starteverdi er 0
    out[strong] = 255    #merker sterke kanter som hvite

    #bruk en kø for å gjøre flood-fill algoritme
    stack = [(rad, kol) for rad, kol in zip(*np.nonzero(strong))]
   
    
    while stack: 
        rad, kol = stack.pop()   #ta ut en piksel fra stack
        for dr in (-1,0,1):     # gå gjennom alle naboer i 8 retninger
            for dk in (-1,0,1):
                if dr == 0 and dk == 0:   # hopp over pikselen selv
                    continue
                rr, kk = rad+dr, kol+dk     #naboens koordinater 
                if 0 <= rr < H and 0<= kk< W:    #sjekk alltid er innenfor bildet
                    if weak[rr,kk] and out[rr,kk] == 0: # sørg for weak kant og har ikke satt inn "out"
                        out[rr,kk] = 255  #inkluder denne svake kanten som en ekte kant
                        stack.append((rr,kk))   # legg den til i stack for videre sjekk
    return out  # returner det endelige kantbildet


def canny_fra_bunnen(gray_img: np.ndarray, sigma = 1, Th = 0.2, Tl = 0.1, normalize_mag = True):
    #1.trinn: Lavpassfiltrer bildet med 2D Gauss-filter
    G = gauss_kjerne(sigma)
    blur = konvolusjon(gray_img, G)
    
    #2.trinn:Finn gradient magnitude og retningen.
    gx,gy,mag,ang = gradient_symmetric(blur)

    #finn maksimal gradientstyrke og normaliser gradientstyrken til [0,1]
    if normalize_mag:
        mmax = mag.max() if mag.max() > 1e-12 else 1.0
        mag = mag/mmax

    mag_nms = nms_fire_retninger(mag,ang)
    kanter = hysteresis(mag_nms, Th,Tl)
    return kanter, (blur,mag,mag_nms)




#leser inn bildet
def read_gray(path):
    """Leser bilde som gråskala uint8.""" 
    img = plt.imread(path)

    # Hvis PNG: float [0,1], skaler til [0,255]
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = img * 255

    # her leser jeg bare geometrimasken inn som et gråtonebilde (fikk denne fra chatgpt)
    if img.ndim == 3:  
        img = img[..., :3].mean(axis=2)  # snitt over RGB

    # Rund av og konverter til uint8
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return img




#oppgave2c
path = "cellekjerner.png"
img = read_gray(path)

sigma = 4
Th = 0.35
Tl = 0.25

edges, (blur, mag, mag_nms)= canny_fra_bunnen(img,sigma, Th,Tl, normalize_mag = True)

save_path = os.path.join(os.path.dirname(path), "detekterte_kanter.png")
plt.imsave(save_path, edges, cmap="gray") #lagre grå bildet
print("Bildet har lagret")

#vis bildet
plt.imshow(edges, cmap="gray")
plt.show() 



#Drøfting av resultatbildet:
#Metoden fungerer bra for å fremheve cellekjernens grenser, men er også lett å finne bakgrunnskanter.
#Resultatet er sterkt avhengig av parameterverdier.
#Ved lav terskelverdi blir flere kjerner tydeligere, men samtidig kommer med mer støy;
#Ved høy terskelverdi blir bakgrunnen renere, men cellekjernene kan bli ufullstendige. 
#Valg av sigma påvirker balasen mellom detaljer og støy: liten sigma beholder mange detaljer men mye støy, mens stor sigma fjerne støy men kan slette små kjerner.
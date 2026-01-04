import numpy as np
import matplotlib.pyplot as plt


# Kontrastjustering

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

def standardize_contrast(img, mu_target=127.0, sigma_target=64.0):
    """Lineær kontrastjustering: nytt mean og std."""
    #gjør om til float64 for å kunne regne med deismaltall verdier er fortsatt [0,255]
    img64 = img.astype(np.float64)

    #gjennomsnitt av gråtoner
    mu_orig = np.mean(img64)
    # standardavvik
    sigma_orig = np.std(img64)
    print("Før justering: mean=", mu_orig," std=", sigma_orig)

    if sigma_orig < 1e-8: #bruker 1e-8 istednfor 0 på grunn av avrundingsfeil i python
        # Unngå deling på null, hvis bildet er helt konstant
        y = np.full_like(img64, mu_target) # lager en array der alle pikslene for verdeien 127 (mu_target)
    else:
        a = sigma_target / sigma_orig
        b = mu_target - a * mu_orig
        print("Koeffisienter: a=", a, ", b=", b)
        y = a * img64 + b
    y = np.clip(np.rint(y), 0, 255).astype(np.uint8)

    mu_new = np.mean(y)
    sigma_new = np.std(y)
    print("Etter justering: mean=", mu_new, ", std=", sigma_new)

    return y


# Geometrijustering

def pick_points_3(img_img, mask_img):
    """Lar deg klikke 3 punkt i kilde, så 3 i maske (rekkefølgen man trykker i er viktig)."""
    #lager en figur med 2 subplot (bildene ved sidena v hverandre)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_img, ax_mask = axes
    # viser bildet og masken i gråskala 
    ax_img.imshow(img_img, cmap="gray");  ax_img.set_title("Kilde: klikk 3 punkt"); ax_img.axis("off")
    ax_mask.imshow(mask_img, cmap="gray"); ax_mask.set_title("Maske: klikk samme 3 punkt i samme rekkefølge"); ax_mask.axis("off")
    plt.tight_layout(); plt.pause(0.1)

    # Kilde
    # velger at vi skal jobbe med bildet først
    plt.sca(ax_img)
    # lar brukeren trykke 3 punkter og returnerer en lsite med 3 punkter
    img_pts = plt.ginput(3, timeout=0)  # [(x1,y1),(x2,y2),(x3,y3)]
    #lager punkter på bildet for å se hvor man har trykket
    ax_img.scatter([p[0] for p in img_pts],[p[1] for p in img_pts], s=40, facecolors="yellow")
    
    # Maske
    plt.sca(ax_mask)
    mask_pts = plt.ginput(3, timeout=0)
    ax_mask.scatter([p[0] for p in mask_pts],[p[1] for p in mask_pts], s=40, facecolors="red")
   

    #tegner prikkene
    plt.draw()
    
    print("Bildepunkter:", img_pts)
    print("Maskepunkter:", mask_pts)

    return np.array(img_pts, dtype=np.float64), np.array(mask_pts, dtype=np.float64)

def solve_affine_3pts(img3: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """
    img3: (3,2) (x,y) - punktene man valgte i pick_points_3
    mask3: (3,2) (X,Y) - punktene man valgte i pick_points_3 
    Returnerer A (2x3) slik at [X Y]^T = A @ [x y 1]^T
    """
    x = img3[:,0]; y = img3[:,1]
    X = mask3[:,0]; Y = mask3[:,1]

    # 6 ukjente: a,b,c,d,e,f
    M = np.zeros((6,6), dtype=np.float64)
    b = np.zeros((6,), dtype=np.float64)

    """
    Vi alger en matrise M for å løse likningene:
    
    M:
    x1 y1 1 0  0  0
    x2 y2 1 0  0  0
    x3 y3 1 0  0  0
    0  0  0 x1 y1 1
    0  0  0 x2 y2 1
    0  0  0 x3 y3 1

    b=[X,Y] altså punktene vi valgte fra masken

    x1,y1,x2 osv. er punktene vi valgte i bildet

    utregningen blir M*[a,b_,c,d,e,f] = b
    
    """
    # X-ligninger
    M[0:3, 0] = x
    M[0:3, 1] = y
    M[0:3, 2] = 1.0
    b[0:3] = X

    # Y-ligninger
    M[3:6, 3] = x
    M[3:6, 4] = y
    M[3:6, 5] = 1.0
    b[3:6] = Y

    params = np.linalg.solve(M, b)  # eksakt for 3 punkt (forutsatt ikke-kollineære)
    a,b_,c,d,e,f = params
    return np.array([[a,b_,c],[d,e,f]], dtype=np.float64)

def invert_affine(A: np.ndarray) -> np.ndarray:
    """
    A (2x3) → H (3x3) → H^-1 → tilbake til (2x3).
    Inversen av matrisen trenger man til baklengs transformasjonen
    """
    #lager en rad til under matrisen slik at det blir 3x3 matrise med 0,0,1 nederst
    H = np.vstack([A, [0.0, 0.0, 1.0]])
    #invers
    Hinv = np.linalg.inv(H)
    return Hinv[:2, :]

def sample_nearest(img: np.ndarray, x: float, y: float) -> float:
    #h=antall rader i y-retning w=antall rader i x-retning
    h, w = img.shape
    #runder av til nærmeste heltall pga. interpolasjon
    xi = int(round(x)); yi = int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return 0.0
    return float(img[yi, xi])

def sample_bilinear(img: np.ndarray, x: float, y: float) -> float:
    """
    funksjonen gjør en billinear interpollasjon for et punkt (x,y) i bildet som er gitt og returnerer farge/ intensiteten til pikselet etterpå
    """
    #h=antall rader i y-retning w=antall rader i x-retning
    h, w = img.shape

    # Finner de fire nærmeste punktene som danner et kvadrat rundt punktet man ser på
    # runder ned
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = x0 + 1; y1 = y0 + 1

    #sjekker siden på bildet der man ikke har 4 punkter
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        #Her kunne jeg også brukt return 0.0, for å få en svart kant. jeg annser det som passende å bruke nearest neighbor helt i kantene da man ikke får nokk punkter å gjøre det billineært.
        return sample_nearest(img, x, y) 

    #hvor mye side- og høydestilte img pikselen er forhold til mask
    dx = x - x0; dy = y - y0
    #I00 og I01 osv. er en intensitetsverdi
    I00 = +float(img[y0, x0])   #øvre venstre
    I10 = float(img[y0, x1])    #øvre høyre
    I01 = float(img[y1, x0])    #nedre venstre
    I11 = float(img[y1, x1])    #nedre høyre
    #justerer intensitet (av en gråtone)
    I0 = I00*(1-dx) + I10*dx                            
    I1 = I01*(1-dx) + I11*dx
    return I0*(1-dy) + I1*dy

def warp_backward(img: np.ndarray, A: np.ndarray, out_h: int, out_w: int, method="bilinear") -> np.ndarray:
    """For hvert mål-piksel (X,Y): finn kilde (x,y)=A^{-1}[X Y 1]^T og sample."""
    Ainv = invert_affine(A)
    #opretter et utbilde i riktig størrelse
    out = np.zeros((out_h, out_w), dtype=np.float64)
    # velger samplemetode utne å kalle metode
    sampler = sample_bilinear if method == "bilinear" else sample_nearest
    #dobbel løkke som går over alle pikslene i bildet
    for Y in range(out_h):
        for X in range(out_w):
            #regner ut hvor pikslene skal være ut ifra den inverse affine transformen
            x = Ainv[0,0]*X + Ainv[0,1]*Y + Ainv[0,2]
            y = Ainv[1,0]*X + Ainv[1,1]*Y + Ainv[1,2]
            #gir de nye posisjonen en intensitet (gråtone)
            out[Y, X] = sampler(img, x, y)

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

def warp_forward(img: np.ndarray, A: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """ Dytter hver kildepiksel til nærmeste målpiksel."""
    #opretter et utbilde i riktig størrelse
    out = np.zeros((out_h, out_w), dtype=np.float64)
    #setter h og w som bredde og høyde fra originalbildet
    h, w = img.shape
    #dobbel for løkke som går gjennom alle punktene
    for y in range(h):
        for x in range(w):
            #gjør den affine transformasjonen på originalbildet for å få nye x og y posisjoner for pisklene
            X = A[0,0]*x + A[0,1]*y + A[0,2]
            Y = A[1,0]*x + A[1,1]*y + A[1,2]
            #piksler trenger heltallsverdier som posisjoner
            Xi = int(round(X)); Yi = int(round(Y))
            if 0 <= Xi < out_w and 0 <= Yi < out_h:
                out[Yi, Xi] = img[y, x]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

def show_image(img, title="Bilde"):
    """Viser et bilde """
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    #Les inn bilde (gråtone) og standardiser kontrast
    img = read_gray("portrett.png")
    out_img = standardize_contrast(img)
    plt.imsave("portrett_kontrast.png", out_img, cmap="gray")
    show_image(out_img, title="Portrett med justert kontrast")
    print("Lagret bilde som portrett_kontrast.png")

    #Les inn maske og plukk 3 punkt i kilde og maske
    mask = read_gray("geometrimaske.png")
    out_h, out_w = mask.shape 
    img_pts, mask_pts = pick_points_3(out_img, mask)

    #Løs affinen fra 3 punktpar (bilde->maske)
    A = solve_affine_3pts(img_pts, mask_pts)
    print("Affin A=\n", A)

    # Baklengs warp både nearest og bilinear
    out_back_near = warp_backward(out_img, A, out_h, out_w, method="nearest")
    plt.imsave("aligned_backward_nearest.png", out_back_near, cmap="gray")
    print("Lagret aligned_backward_nearest.png")

    out_back_bi = warp_backward(out_img, A, out_h, out_w, method="bilinear")
    plt.imsave("aligned_backward_bilinear.png", out_back_bi, cmap="gray")
    print("Lagret aligned_backward_bilinear.png")

    # Forlengs warpogså kalt naiv push
    out_forw = warp_forward(out_img, A, out_h, out_w)
    plt.imsave("aligned_forward_naive.png", out_forw, cmap="gray")
    print("Lagret aligned_forward_naive.png")

    # Valgfritt: vis raskt
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original (før kontrast)")
    axes[0].axis("off")

    axes[1].imshow(out_img, cmap="gray")
    axes[1].set_title("Etter kontrastjustering")
    axes[1].axis("off")

    axes[2].imshow(out_back_near, cmap="gray")
    axes[2].set_title("Baklengs – nearest")
    axes[2].axis("off")

    axes[3].imshow(out_back_bi, cmap="gray")
    axes[3].set_title("Baklengs – bilinear")
    axes[3].axis("off")

    axes[4].imshow(out_forw, cmap="gray")
    axes[4].set_title("Forlengs – push")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
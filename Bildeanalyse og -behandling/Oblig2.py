import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import fft as sfft
from time import perf_counter


img = mpimg.imread("cow.png").astype(np.float64)
if img.ndim != 2:
    raise ValueError("Forventer et gråtonebilde (2D).")
H, W = img.shape

kernel = np.ones((15, 15), dtype=np.float64) / (15 * 15)

filtered_fill = signal.convolve2d(img, kernel, mode='same', boundary='fill')
filtered_wrap = signal.convolve2d(img, kernel, mode='same', boundary='wrap')

def fft_wrap(img, kernel):
    H, W = img.shape # høyde og bredde på bildet
    kh, kw = kernel.shape # høyde og bredde på filterkjernen

    # PSF = Point Spread Function 
    psf = np.zeros((H, W), dtype=np.float64) # Lager et 0-bilde på samme størrelse
    psf[:kh, :kw] = kernel # Setter inn filterkjernen øverst til venstre

    # Flytt kjernens senter til (0,0) for korrekt sirkulær konvolusjon
    psf = np.roll(psf, -kh // 2, axis=0)# flytter i høyderetning
    psf = np.roll(psf, -kw // 2, axis=1) # flytter i bredderetning

    # FFT av bilde og filter
    img_fft = sfft.fft2(img) # Fourier-transformasjon av bildet. bruker sitt eget bilde som wrap
    psf_fft = sfft.fft2(psf) # Fourier-transformasjon av filteret (PSF)

    filtered = sfft.ifft2(img_fft * psf_fft).real # Multipliser i frekvensdomenet og ta invers FFT

    return filtered


def fft_fill(img, kernel):
    H, W = img.shape # høyde og bredde på bildet
    kh, kw = kernel.shape # høyde og bredde på filterkjernen

    FH, FW = H + kh - 1, W + kw - 1 #legger til kjerne breddeg høyde slik at når kjernen er i ytterst vil deen fortsatt være innenfor deifnisjonen av bildet (selvom dette blir satt som svart)
    img_fft = sfft.fft2(img, (FH, FW)) #når man bruker fft2 og gir verdier til s betyr det at man gir bildet en størrelse. hvis størrelsene er større enn selve bildet gir det en padding med 0-er rundt der bildet ikke er angitt.
    ker_fft = sfft.fft2(kernel, (FH, FW))
    conv_full = sfft.ifft2(img_fft * ker_fft).real #multipliserer  og tar inversen. bildet har nå fortsatt en padding

    kernel_senter_h, kernel_senter_w = kh // 2, kw // 2 #måler bredde og høyde av senter av kjernen
    conv_same = conv_full[kernel_senter_h:kernel_senter_h+H, kernel_senter_w:kernel_senter_w+W] # sier hvor vi skal klippe for å ikkeha med padding. øverst til venstre til nederst til høyre
    return conv_same



fft_filtered_wrap = fft_wrap(img, kernel)
fft_filtered_fill = fft_fill(img, kernel)

plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')

plt.subplot(2, 3, 2); plt.imshow(filtered_wrap, cmap='gray'); plt.title("Spatial wrap"); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(fft_filtered_wrap, cmap='gray'); plt.title("FFT wrap"); plt.axis('off')

plt.subplot(2, 3, 5); plt.imshow(filtered_fill, cmap='gray'); plt.title("Spatial fill"); plt.axis('off')
plt.subplot(2, 3, 6); plt.imshow(fft_filtered_fill, cmap='gray'); plt.title("FFT fill"); plt.axis('off')

plt.tight_layout()
plt.show()

# oppgave 2

def timeit_many(func, repeats=25):
    times = []
    func()  # liten warmup
    for _ in range(repeats):
        t0 = perf_counter() #starttid
        _ = func()
        t1 = perf_counter() #sluttid
        times.append(t1 - t0) #tid
    arr = np.array(times, dtype=np.float64) #legger tallene inn i numpy array slik at man kan gjøre mean og std på dem
    return float(arr.mean()), float(arr.std(ddof=1)) #returnereer mean og std

kernel_sizes = [3, 5, 7, 15, 31, 63]
repeats = 25 

mean_spatial_fill, std_spatial_fill = [], []
mean_fft_fill,     std_fft_fill     = [], []

mean_spatial_wrap, std_spatial_wrap = [], []
mean_fft_wrap,     std_fft_wrap     = [], []

for k in kernel_sizes: #går gjenngom alle kjerne størrelser
    ker = np.ones((k, k), dtype=np.float64) / (k * k) #lager kjernen

    m, s = timeit_many(lambda: signal.convolve2d(img, ker, mode='same', boundary='fill'), #konvolusjon fill
                       repeats=repeats)
    mean_spatial_fill.append(m); std_spatial_fill.append(s)

    m, s = timeit_many(lambda: fft_fill(img, ker), repeats=repeats) #fft fill
    mean_fft_fill.append(m); std_fft_fill.append(s)

    m, s = timeit_many(lambda: signal.convolve2d(img, ker, mode='same', boundary='wrap'), #kovolusjon wrap
                       repeats=repeats)
    mean_spatial_wrap.append(m); std_spatial_wrap.append(s)

    m, s = timeit_many(lambda: fft_wrap(img, ker), repeats=repeats) # fft wrap
    mean_fft_wrap.append(m); std_fft_wrap.append(s)

plt.figure(figsize=(10,5))

# venstre side (fill)
plt.subplot(1,2,1)
plt.plot(kernel_sizes, mean_spatial_fill, 'o-', label='konvolusjon') #konvolusjonsgraf
plt.plot(kernel_sizes, mean_fft_fill, 's-', label='fft') #ftt-graf
plt.title("fill")
plt.xlabel("kjerne størrelse (k x k)")
plt.ylabel("tid [s]")
plt.legend()
plt.grid(True)

# høyre side (wrap)
plt.subplot(1,2,2)
plt.plot(kernel_sizes, mean_spatial_wrap, 'o-', label='konvolusjon') #kovolusjonsgraf
plt.plot(kernel_sizes, mean_fft_wrap, 's-', label='fft') #fft-graf
plt.title("wrap")
plt.xlabel("kjerne størrelse (k x k)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\nGjennomsnittstid [ms] per metode (fill):")
for i, (k, s_mean, f_mean) in enumerate(zip(kernel_sizes, mean_spatial_fill, mean_fft_fill)): # i er en teller som øker med 1 hve rgange, mens k, s_mean, f_mean får hver sin verdi fra listene kernel_sizes, mean_spatial_wrap, mean_fft_wrap
    print("k =", k, "konvolusjon:", round(s_mean*1e3, 3), "ms ±", round(std_spatial_fill[i]*1e3, 2),
          "--- fft:", round(f_mean*1e3, 3), "ms ±", round(std_fft_fill[i]*1e3, 2))

print("\nGjennomsnittstid [ms] per metode (wrap):")
for i, (k, s_mean, f_mean) in enumerate(zip(kernel_sizes, mean_spatial_wrap, mean_fft_wrap)): # i er en teller som øker med 1 hve rgange, mens k, s_mean, f_mean får hver sin verdi fra listene kernel_sizes, mean_spatial_wrap, mean_fft_wrap
    print("k =", k, "konvolusjon:", round(s_mean*1e3, 3), "ms ±", round(std_spatial_wrap[i]*1e3, 2),
          "--- fft:", round(f_mean*1e3, 3), "ms ±", round(std_fft_wrap[i]*1e3, 2))
    

#oppgave 3

def ideal_lowpass(img, radius_pix):
    H, W = img.shape
    F = sfft.fft2(img) #frekvenser av bildet
    F = sfft.fftshift(F) #nullfrekvens i sentrum (shifted quadrants)

    #lager sirkulær maske 
    #maske laging fikk jeg litt hjelp fra chatgpt til å lage
    yy, xx = np.ogrid[-H//2:H//2, -W//2:W//2] #lager en grid osm er samme størrelse som bilde, men der (0,0) er sentrum
    rr = np.sqrt(xx*xx + yy*yy) #gir hver verdi i griden et tall som er avstanden til sentrum. Dette brukes til å kunne angi en avstand fra sentrum man vil se frekvens verdier for.
    mask = (rr <= radius_pix).astype(np.float64) #lager en sirkulær maske der man setter verdier til 1 eller 0 utifra om de er innenfor den gitte avstanden. 

    F_lp = F * mask #overalt hvor det stod 0 i masken vil det også stå 0 i frekvensdomenet
    out = sfft.ifft2(sfft.ifftshift(F_lp)).real #inverstransform for å få tilbake bilde med de nye frekvensene (blir mer riktig å sik at det er de samme frekvensene, men at de høye frekvensene utenfor sirkelen er tatt bort)
    return out

radius = int(round(min(H, W) / 20))  #radius til sirkelen (delt på 20 gir ca 5% av frekvensene)
ideal_lp_img = ideal_lowpass(img, radius)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(filtered_fill, cmap='gray'); plt.title("15×15 (fill)"); plt.axis('off') #kovolusjon fill
plt.subplot(1,2,2); plt.imshow(ideal_lp_img, cmap='gray'); plt.title(f"Ideelt LP (radius={radius})"); plt.axis('off')
plt.tight_layout(); plt.show()

#spekter for 15x15 filter
psf_box = np.zeros_like(img) #lager et svart bilde på samme størrelse som inn bilde
psf_box[:15, :15] = kernel #setter inn kjerne øverst til venstre
psf_box = np.roll(psf_box, -15//2, axis=0) #flytter kjerne til midten
psf_box = np.roll(psf_box, -15//2, axis=1) #flytter kjerne til midten
F_box = sfft.fftshift(sfft.fft2(psf_box)) #setter 0 frekvens i midten etter å ha tatt FFT
mag_box = np.abs(F_box)  #må ta absolutt verdien for å se bort ifra imaginære verdier

# maske for lavpassfilter = spekter av filter
yy, xx = np.ogrid[-H//2:H//2, -W//2:W//2]
rr = np.sqrt(xx*xx + yy*yy)
mask = (rr <= radius).astype(np.float64)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(mag_box, cmap='gray'); plt.title("Spekter: 15×15-boks"); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(mask, cmap='gray'); plt.title(f"Ideelt LP (maske, r={radius})"); plt.axis('off')
plt.tight_layout(); plt.show()
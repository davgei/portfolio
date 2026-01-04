import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;
import java.util.concurrent.CountDownLatch;

public class Oblig2Del2B {

static final int fletteTraadAntall = 8;
static CountDownLatch latch = new CountDownLatch(fletteTraadAntall);

    public static void main(String[] args) {

        //Dette gjør slik at du kan sende inn en parameter i terminalen. altså mappenavnet er parameteren

        if (args.length != 1) {
            System.out.println("Skriv slik for starte koden: java Oblig2Del2 <mappe>");
            return;
        }

        String mappeNavn = args[0];
        File metadataFil = new File(mappeNavn + "/metadata.csv");


        if (!metadataFil.exists()) {
            System.out.println("Metadatafil ikke funnet i mappen: " + mappeNavn);
            return;
        }

        try {

            Monitor2 monitor = new Monitor2();

            // Leser filer og legger dem til monitoren
            Thread[] threads = startLeseTraader(monitor, metadataFil, mappeNavn);

            for (Thread thread : threads) {
                thread.join();
            }

            // Fletter alle HashMap-ene i Monitoret
            HashMap<String, Subsekvens> flettetMap = flettHashMaps(monitor);
            // Skriver ut subsekvensen med flest forekomster
            skrivUtMaksForekomster(flettetMap);
        } catch (InterruptedException  e) {
            System.out.println("En feil oppstod under utførelsen av tråder: " + e.getMessage());

        } catch (FileNotFoundException e) {

            System.out.println("Feil ved lesing av metadatafil: " + e.getMessage());
        }
    }

    // Leser filene i mappen og legger dem til monitoren
    public static Thread[] startLeseTraader(Monitor2 monitor, File metadataFil, String mappeNavn) throws FileNotFoundException {
        Scanner scanner = new Scanner(metadataFil);

        // Leser metadatafilen for å telle antall linjer (filer) for å vite hvor mange threads
        int numLines = 0;
        Scanner metadataScanner = new Scanner(metadataFil);
        while (metadataScanner.hasNextLine()) {
            numLines++;
            metadataScanner.nextLine();
        }
        metadataScanner.close();

        // Oppretter en trådarray med riktig størrelse basert på antall linjer (filer i metadata)
        Thread[] threads = new Thread[numLines];

        int i = 0;
        while (scanner.hasNextLine()) {
            String filNavn = mappeNavn+"/"+scanner.nextLine();
            File fil = new File(filNavn);

            if (fil.exists()) {
                Thread thread = new LeseTrad(filNavn, monitor);
                threads[i++] = thread;
                thread.start();
            } else {
                System.out.println("Fil ikke funnet: " + filNavn);
            }
        }
        scanner.close();
        return threads;
    }

    // Fletter alle HashMap-ene i monitoret
    public static HashMap<String, Subsekvens> flettHashMaps(Monitor2 monitor) {
        Thread[] traader = new Thread[fletteTraadAntall];
        for (int i = 0; i < fletteTraadAntall; i++) {
            traader[i] = new FletteTrad(monitor, latch);
            traader[i].start();
        }

        //passer på at koden først kjører videre etter at alle threadsene er ferdige med å kjøre
        try {
            latch.await();
        } catch (InterruptedException e) {
            // Handle InterruptedException
            e.printStackTrace(); 
        }
        return monitor.hentVilkaarligMap();
    }

    // Skriver ut subsekvensen med flest forekomster
    public static void skrivUtMaksForekomster(HashMap<String, Subsekvens> map) {
        String maksSubsekvens = "";
        int maksForekomster = 0;
        for (String subsekvens : map.keySet()) {
            int forekomster = map.get(subsekvens).hentAntall();
            if (forekomster > maksForekomster) {
                maksSubsekvens = subsekvens;
                maksForekomster = forekomster;
            }
        }
        System.out.println("Subsekvens med flest forekomster: " + maksSubsekvens + " med " + maksForekomster + " forekomster");
    }
}
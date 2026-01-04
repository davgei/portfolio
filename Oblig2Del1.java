import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;

public class Oblig2Del1 {
    public static void main(String[] args) {

        if (args.length != 1) {
            System.out.println("Skriv slik for starte koden: java Oblig2Del1 <mappe>");
            return;
        }

        String mappeNavn = args[0];
        File metadataFil = new File(mappeNavn + "/metadata.csv");


        if (!metadataFil.exists()) {
            System.out.println("Metadatafil ikke funnet i mappen: " + mappeNavn);
            return;
        }

        try {

            SubsekvensRegister subsekvensRegister = new SubsekvensRegister();

            // Leser filer og legger dem til SubsekvensRegisteret
            lesOgLeggTil(subsekvensRegister, metadataFil, mappeNavn);

            // Fletter alle HashMap-ene i SubsekvensRegisteret
            HashMap<String, Subsekvens> flettetMap = flettHashMaps(subsekvensRegister);
            // Skriver ut subsekvensen med flest forekomster
            skrivUtMaksForekomster(flettetMap);
        } catch (FileNotFoundException e) {
            System.out.println("Feil ved lesing av metadatafil: " + e.getMessage());
        }
    }

    // Leser filene i mappen og legger dem til SubsekvensRegisteret
    public static void lesOgLeggTil(SubsekvensRegister subsekvensRegister, File metadataFil, String mappeNavn) throws FileNotFoundException {
        Scanner scanner = new Scanner(metadataFil);
        while (scanner.hasNextLine()) {
            String filNavn = mappeNavn+"/"+scanner.nextLine();
            File fil = new File(filNavn);
            if (fil.exists()) {
                HashMap<String, Subsekvens> hashMap = SubsekvensRegister.lesFraFil(filNavn);
                subsekvensRegister.leggTilHashMap(hashMap);
            } else {
                System.out.println("Fil ikke funnet: " + filNavn);
            }
        }
        scanner.close();
    }

    // Fletter alle HashMap-ene i SubsekvensRegisteret
    public static HashMap<String, Subsekvens> flettHashMaps(SubsekvensRegister subsekvensRegister) {
        while (subsekvensRegister.hentRegisterStorrelse() > 1) {
            HashMap<String, Subsekvens> map1 = subsekvensRegister.hentVilkaarligMap();
            HashMap<String, Subsekvens> map2 = subsekvensRegister.hentVilkaarligMap();
            //hadde problem med at den tok samme map og laseg selv til seg selv slik at man bare fkk en dobling av mapet
            while(map1==map2){
                map2 = subsekvensRegister.hentVilkaarligMap();
            }
            HashMap<String, Subsekvens> flettetMap = SubsekvensRegister.slaaSammenMaps(map1, map2);
            subsekvensRegister.leggTilHashMap(flettetMap);

        }

        return subsekvensRegister.hentVilkaarligMap();
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
import java.util.*; //tror egentlig at denne importerer alt fra util
import java.io.File;
import java.io.FileNotFoundException;


public class SubsekvensRegister {
    private ArrayList<HashMap<String,Subsekvens>> subsekvensRegister;

    public SubsekvensRegister() {
        this.subsekvensRegister = new ArrayList<>();
    }
    
    public void leggTilHashMap(HashMap<String, Subsekvens> hashMap){
        subsekvensRegister.add(hashMap);
    }

    public HashMap<String, Subsekvens> hentVilkaarligMap(){
        if (subsekvensRegister.size()<1) {
            return null;
        } else{
            return subsekvensRegister.remove(0);
        }
    }

    public HashMap<String, Subsekvens> hentFlettetMap(){
        //fjerner ikke flettet map når man viil hente det ut. Fint for å bare lese mapet
        return subsekvensRegister.get(0);
    }



    public int hentRegisterStorrelse(){
        return subsekvensRegister.size();
    }

    public static HashMap<String, Subsekvens> lesFraFil(String filNavn) {
        HashMap<String, Subsekvens> hashMap = new HashMap<>();

        File fil = new File(filNavn);

        try {
            Scanner scanner = new Scanner(fil);

            while (scanner.hasNextLine()) {
                String linje = scanner.nextLine();
                if (linje.length() < 3) {
                    // Avslutter while-løkken hvis linjen er kortere enn 3 tegn
                    break;
                }
                for (int i = 0; i <= linje.length() - 3; i++) {
                    String subsekvens = linje.substring(i, i + 3);
                    if (!hashMap.containsKey(subsekvens)) {
                        // Legg til subsekvensen i HashMap-en med antall 1
                        hashMap.put(subsekvens, new Subsekvens(subsekvens, 1));
                    }
                }
            }

            scanner.close();
        } catch (FileNotFoundException e) {
            System.out.println("Filen ble ikke funnet: " + filNavn);
            System.exit(-1);
        }

        return hashMap;
    }

    static HashMap<String, Subsekvens> slaaSammenMaps(HashMap<String, Subsekvens> map1, HashMap<String, Subsekvens> map2){
        HashMap<String, Subsekvens> sammenflettetMap = new HashMap<>();
        
        //legger map1 inn i det nye mapet
        for (String nokkel : map1.keySet()) {
            Subsekvens subsekvens1 = map1.get(nokkel);
            sammenflettetMap.put(nokkel, subsekvens1);
        }

        //sammenligner map 2 mot map1 om subskevensene allerede finnes eller ikke. Hvis definnes øker bare tallet i subsekvensen
        for(String nokkel : map2.keySet()){
            if (sammenflettetMap.containsKey(nokkel) ) {
                Subsekvens subsekvens1 = sammenflettetMap.get(nokkel);
                subsekvens1.endreAntall(subsekvens1.hentAntall()+map2.get(nokkel).hentAntall());
            } else {
                Subsekvens subsekvens2 = map2.get(nokkel);
                sammenflettetMap.put(nokkel, subsekvens2);
            }
        }
        
        return sammenflettetMap;
    }



}

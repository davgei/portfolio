import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CountDownLatch;

public class FletteTrad extends Thread{
    private Monitor2 monitor;
    private final CountDownLatch latch;


    public FletteTrad(Monitor2 monitor, CountDownLatch latch) {
        this.monitor = monitor;
        this.latch = latch;

    }

    
    public void run() {
        while (true) {
        
            List<HashMap<String, Subsekvens>> hashMapListe = null;
            hashMapListe = monitor.hentUtTo();
            if(hashMapListe == null){
                latch.countDown();
                break;
            }
        
            // Sjekk om listen har minst to HashMaps
            HashMap<String, Subsekvens> map1 = hashMapListe.get(0);
            HashMap<String, Subsekvens> map2 = hashMapListe.get(1);

            if (map1 == null || map2 == null) {
                latch.countDown();
                break;
            }
            // Flett de to HashMap-ene
            HashMap<String, Subsekvens> flettetMap = SubsekvensRegister.slaaSammenMaps(map1, map2);
            // Legg til det flettede HashMap-et i monitoren
            monitor.leggTilHashMap(flettetMap);  
        
                 
        }
    }
}

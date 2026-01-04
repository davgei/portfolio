import java.util.*;

public class Monitor2 {
    private final SubsekvensRegister subsekvensRegister;
    private final Object lock = new Object();
    private int threadsVenter = 0;
    private boolean sisteFerdig = false;


    public Monitor2() {
        this.subsekvensRegister = new SubsekvensRegister();
    }

    public  void leggTilHashMap(HashMap<String, Subsekvens> hashMap) {
        synchronized (lock){
            subsekvensRegister.leggTilHashMap(hashMap);
            lock.notify();
        }
    }

    public synchronized HashMap<String, Subsekvens> hentVilkaarligMap() {
        return subsekvensRegister.hentVilkaarligMap();
    }

    public int hentRegisterStorrelse() {
        return subsekvensRegister.hentRegisterStorrelse();
    }

    public synchronized HashMap<String, Subsekvens> hentFlettetMap() {
        return subsekvensRegister.hentFlettetMap();
    }

    public static HashMap<String, Subsekvens> lesFraFil(String filNavn) {
        return SubsekvensRegister.lesFraFil(filNavn);
    }

    public static  HashMap<String, Subsekvens> slaaSammenMaps(HashMap<String, Subsekvens> map1, HashMap<String, Subsekvens> map2) {
        return SubsekvensRegister.slaaSammenMaps(map1, map2);
    }

    

    public List<HashMap<String, Subsekvens>> hentUtTo(){
        synchronized (lock){
            if (hentRegisterStorrelse()<2) {
                if(sisteFerdig==true){
                    return null;
                }
            }
            List<HashMap<String, Subsekvens>> toHashMaps;
            
            while(true){      
                if(this.hentRegisterStorrelse() < 2){
                    try{
                        if(threadsVenter <= 7){
                            threadsVenter++;
                            lock.wait(); // Vent til det er minst to HashMap-er i registeret
                            threadsVenter--;
                            if(hentRegisterStorrelse()<2){
                                sisteFerdig = true;
                                lock.notifyAll();
                                return null;
                            } else {
                                break;
                            }
                        }  else{
                            sisteFerdig = true;
                            lock.notifyAll();
                            return null;
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace(); 
                        Thread.currentThread().interrupt();
                    }
                
                } else {
                    break;
                }
            }
            toHashMaps = new ArrayList<>();
            HashMap<String, Subsekvens> map1 = this.hentVilkaarligMap();
            HashMap<String, Subsekvens> map2 = this.hentVilkaarligMap();
           
                while(map1==map2){
                    map2 = this.hentVilkaarligMap();
                }
            toHashMaps.add(map1);
            toHashMaps.add(map2);

            // Returner listen med de to HashMaps
            return toHashMaps;
        }
    } 
}

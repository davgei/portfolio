import java.util.HashMap;

public class LeseTrad extends Thread {
    
    private String filNavn;
    private Monitor2 monitor;

    public LeseTrad(String filNavn, Monitor2 monitor) {

        this.filNavn = filNavn;
        this.monitor = monitor;
    }
    
    //leser fra fil og legger det tilsom hashmap i monitoren
    public void run(){
        HashMap<String, Subsekvens> hashmap = Monitor2.lesFraFil(this.filNavn);
        monitor.leggTilHashMap(hashmap);
    }
}

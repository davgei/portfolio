public class Subsekvens {
    private final String subsekvens;
    private int antall;

    public Subsekvens(String subsekvens, int antall) {
        this.subsekvens = subsekvens;
        this.antall = antall;
    }

    public String hentSubsekvens() {
        return subsekvens;
    }

    public int hentAntall() {
        return antall;
    }

    public void endreAntall(int antall) {
        this.antall = antall;
    }

    @Override
    public String toString() {
        return "(" + subsekvens + "," + antall + ")";
    }
}
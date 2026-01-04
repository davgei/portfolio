.text
.global main

main:
    MOV r0, #13        @ Sett N = 13, du kan endre dette for å teste andre verdier
    MOV r1, #0         @ current, Fibonacci F(0)
    MOV r2, #1         @ previous, Fibonacci F(1)

    CMP r0, #0         @ Sjekk om N er 0
    BEQ exit           @ Hvis N er 0, skriv ut current (0) og avslutt

loop:
    SUB r0, r0, #1     @ Reduser N med 1
    MOV r3, r1         @ Midlertidig lagring for den nåværende verdien
    ADD r1, r2, r1     @ current = previous + current
    MOV r2, r3         @ Oppdater previous til den gamle nåværende verdien

    CMP r0, #0         @ Sammenlign N med 1
    BEQ exit
    B loop             @ Hopp til starten av løkken

exit:
    MOV r0, r1         @ Flytt resultatet til r0 for retur
    BX lr              @ Returner fra funksjonen

.section .note.GNU-stack,"",%progbits

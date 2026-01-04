.global main

fib:
    PUSH {r4, lr}             @ Dytt r4 og link register (returadresse) til stacken
    MOV r3, #1                @ Initialiser previous til 1
    MOV r2, #0                @ Initialiser current til 0
    CMP r0, #0                @ Sammenlign N med 0
    BEQ zero_case             @ Hvis N == 0, håndter spesialtilfellet



fib_loop:
    SUBS r0, r0, #1           @ Reduser N med 1
    MOV r4, r2                @ Lagre current i r4
    ADD r2, r2, r3            @ current = current + previous
    MOV r3, r4                @ previous = gammel current
    BEQ end_loop              @ Hvis N er 0, er vi ferdige
    B fib_loop                @ Fortsett løkken

zero_case:
    MOV r0, #0                @ Resultatet er 0 når N=0
    B end_fib                 @ Hopp til avslutning


end_loop:
    MOV r0, r2                @ Flytt current (resultatet) til r0


end_fib:
    POP {r4, lr}              @ Gjenopprett r4 og returner til den som kalte funksjonen
    BX lr


main:
    PUSH {lr}
    LDR r0, =13               @ Last ønsket Fibonacci-nummer inn i r0
    BL fib                    @ Kall fib-funksjonen
    MOV r1, r0                @ Flytt resultatet til r1 for printf
    LDR r0, =output_string    @ Last adressen til output-formatstrengen
    BL printf                 @ Kall printf for å skrive ut resultatet
    MOV r0, #0
    POP {lr}
    BX lr                     @ Returner fra main

.data
output_string:
    .asciz "%d\n"
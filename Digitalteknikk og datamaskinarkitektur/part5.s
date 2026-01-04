    .data
    num1:       .word   0x40400000     @ Flyttall 1.0 i IEEE 754
    num2:       .word   0x40600000     @ Flyttall 2.0 i IEEE 754
    exp_mask:   .word   0x7F800000     @ Maske for å hente ut eksponenten
    mant_mask:  .word   0x007FFFFF     @ Maske for å hente ut mantissa

    .text
    .global main
main:
    LDR r0, =num1            @ Laster adressen til num1
    LDR r1, [r0]             @ r1 får verdien til num1 (1.0 i IEEE 754)
    LDR r0, =num2            @ Laster adressen til num2
    LDR r2, [r0]             @ r2 får verdien til num2 (2.0 i IEEE 754)

    LDR r3, =exp_mask        @ Laster eksponentmasken
    LDR r3, [r3]
    AND r4, r1, r3           @ Bruker masken for å hente ut eksponenten til num1
    AND r5, r2, r3           @ Henter ut eksponenten til num2
    LSR r4, r4, #23          @ Flytter eksponenten til riktig posisjon og biten foran vil ikke gjøre noe fordid en alltid vil bli 0
    LSR r5, r5, #23         

    LDR r3, =mant_mask       @ Laster mantissamasken
    LDR r3, [r3]
    AND r6, r1, r3           @ Bruker masken for å hente ut mantissaen til num1
    AND r7, r2, r3           @ Henter ut mantissaen til num2
    ORR r6, r6, #0x00800000  @ Legger til ledende 1 i mantissaen 
    ORR r7, r7, #0x00800000  

    CMP r4, r5               @ Sjekker hvilken eksponent som er størst (for å vite hvilken matissa soms kal shiftes)
    BGT adjust_r7            @ Hvis r4 > r5 så hopper kdoen til adjust_r7 for å justere r7
    SUB r8, r5, r4           @ Finn differansen mellom eksponentene (hvis r5 > r4)
    LSR r6, r6, r8           @ Justerer mantissaen til num1 (flytter til høyre)
    MOV r4, r5               @ Setter r4 til den største eksponenten (trengs for hele tallet etterpå)
    B add_mantissas
adjust_r7:
    SUB r8, r4, r5           @ Finn differansen mellom eksponentene (hvis r4 > r5)
    LSR r7, r7, r8           @ Justerer mantissaen til num2 (flytter til høyre)
    B add_mantissas	     @ Hopper til add mantissas slik at man ikke også subtraherer andre veien og flytter den andre matissaen

add_mantissas:
    ADD r6, r6, r7           @ Legger sammen mantissaene

    TST r6, #0x01000000      @ Sjekker om mantissaen ble for stor (overflow)
    LSRNE r6, r6, #1         @ Hvis overflow, normaliser ved å flytte mantissaen til høyre
    ADDNE r4, r4, #1         @ Øker eksponenten hvis det var overflow

    BIC r6, r6, #0x00800000  @ Fjerner det ledende 1-tallet fra mantissaen
    LSL r4, r4, #23          @ Flytter eksponenten tilbake til riktig posisjon
    ORR r0, r4, r6           @ Kombinerer eksponent og mantissa til det endelige tallet

    BX lr                    @ Returner fra funksjonen

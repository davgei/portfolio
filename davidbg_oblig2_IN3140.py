import numpy as np

# Lenkelengder (mm):
L1 = 100.9
L2 = 222.1
L3 = 136.2

#1A

def forward(joint_angles_degs):
    """
    Forward kinematics for a 3-DOF (shoulder-elbow) robot.
    Input:
      joint_angles_degs = [theta1, theta2, theta3] in degrees
    Output:
      cart_cord = np.array([x, y, z]) in mm (or same unit as L1, L2, L3)
    """

    # Hent leddvinklene (grader):
    theta1_deg, theta2_deg, theta3_deg = joint_angles_degs

    # Konverter til radianer:
    t1 = np.deg2rad(theta1_deg)
    t2 = np.deg2rad(theta2_deg)
    t3 = np.deg2rad(theta3_deg)

    # Forhåndsberegn noen trigonometriske uttrykk:
    c1, s1 = np.cos(t1), np.sin(t1)
    c2, s2 = np.cos(t2), np.sin(t2)
    c23 = np.cos(t2 + t3)
    s23 = np.sin(t2 + t3)

    # Standard formel for 3-ledd manipulator:
    x = L2 * c1 * c2      + L3 * c1 * c23
    y = L2 * s1 * c2      + L3 * s1 * c23
    z = L1 + L2 * s2      + L3 * s23

    # Returner kartesiske koordinater som en numpy-array (x, y, z):
    return np.round(np.array([x, y, z]), 4)




#1B
def inverse(cart_cord):
    """
    Inverse kinematics for a 3-DOF (shoulder-elbow) robot arm.
    Input:
      cart_cord = [x, y, z]  (float)
    Output:
      joint_angles_degs = [theta1, theta2, theta3] in degrees (float)

    Assumes:
      - 'elbow up' solution
      - link lengths: L1, L2, L3
    """

    # 1) Hent ut (x,y,z)
    x, y, z = cart_cord

    # 3) Beregn "r" (horisontal avstand) og z' (vertikal ift. skulder)
    r = np.sqrt(x**2 + y**2)
    z_prime = z - L1

    # 4) Avstand a fra skulderen til tupp
    a = np.sqrt(r**2 + z_prime**2)

    # 5) Finn theta3 via cosinusloven:
    #    cos(theta3) = (a^2 - L2^2 - L3^2) / (2 * L2 * L3)
    cos_th3 = (a**2 - L2**2 - L3**2) / (2 * L2 * L3)
    # For å unngå numeriske feil utenfor [-1,1]:
    cos_th3 = np.clip(cos_th3, -1.0, 1.0)

    # "Albue opp"-løsning:
    th3 = np.arccos(cos_th3)   # vinkel i rad

    # 6) Finn theta2
    #    Vi bruker standard “split” form:
    #        phi = atan2(z', r)
    #        alpha = atan2(L3 * sin(th3), L2 + L3 * cos(th3))
    #        theta2 = phi - alpha
    phi = np.arctan2(z_prime, r)
    alpha = np.arctan2(L3 * np.sin(th3), L2 + L3 * np.cos(th3))
    th2 = phi - alpha

    # 7) Finn theta1 = rotasjon i horisontalplanet
    th1 = np.arctan2(y, x)

    # 8) Konverter fra radianer til grader
    deg1 = np.rad2deg(th1)
    deg2 = np.rad2deg(th2)
    deg3 = np.rad2deg(th3)

    # Returner i en numpy-array
    return np.round(np.array([deg1, deg2, deg3]),4) 

#1C
def verify_forward_inverse():
    # 1) Et sett leddvinkler
    joint_angles = [270, -30, 45]  

    # 2) Kjør forward -> posisjon
    cart_pos = forward(joint_angles)

    # 3) Kjør inverse -> nye leddvinkler
    new_angles = inverse(cart_pos)

    # 4) Kjør forward igjen -> pos_2
    cart_pos_2 = forward(new_angles)

    # 5) Rund av til 4 desimaler
    cp1 = [round(v, 4) for v in cart_pos]
    cp2 = [round(v, 4) for v in cart_pos_2]

    # 6) Sjekk differanser (x, y, z)
    dx = abs(cp1[0] - cp2[0])
    dy = abs(cp1[1] - cp2[1])
    dz = abs(cp1[2] - cp2[2])

    # 7) Toleranse-test
    if dx < 0.0001 and dy < 0.0001 and dz < 0.0001:
        print("True - forward/inverse kinematics match up.")
    else:
        print("False - there is a mismatch.")

verify_forward_inverse()




#1D
#Jeg gjorde litt om på forrige inverse funksjonen slik at den skriver ut de 4 mulige settene. Likninga ha rjeg fra forrige oblig.

def find_four_solutions():
    """
    Viser 4 IK-løsninger slik brukeren spesifikt ønsker:
      1) Høyrevendt, albue opp
      2) Høyrevendt, albue ned
      3) Venstrevendt, albue opp (theta1+180°, theta2+180°)
      4) Venstrevendt, albue ned
    For en 3-ledd robot:
       L1=100.9, L2=222.1, L3=136.2
    Posisjon:
       (x,y,z) = (0, -323.9033, 176.6988)

    NB:
    - Dette er ikke "standard" lærebok-løsning, men EXACTly 
      hva du beskrev: +180° på både theta1 og theta2 for venstre side.
    """

    # 1) Kjent geometri + TCP-pos
    L1, L2, L3 = 100.9, 222.1, 136.2
    x, y, z    = 0.0, -323.9033, 176.6988

    # 2) r, z', a
    r = np.sqrt(x**2 + y**2)     # ~323.9033
    zprime = z - L1             # ~75.7988
    a = np.sqrt(r**2 + zprime**2)

    # 3) cos(theta3) via cosinusloven
    cos_th3 = (a**2 - L2**2 - L3**2)/(2*L2*L3)
    cos_th3 = np.clip(cos_th3, -1, 1)

    # "albue opp" => +arccos ; "albue ned" => -arccos
    th3_up = np.arccos(cos_th3)
    th3_down = -th3_up

    # 4) Finn phi = arctan2(z', r) + alpha for skulder
    phi = np.arctan2(zprime, r)

    def alpha_for(th3):
        return np.arctan2(L3*np.sin(th3), L2 + L3*np.cos(th3))

    alpha_up   = alpha_for(th3_up)
    alpha_down = alpha_for(th3_down)

    # => skulderledd
    th2_up   = phi - alpha_up
    th2_down = phi - alpha_down

    # => baseledd
    #   arctan2(y,x) => ~-90° i rad
    theta1_right = np.arctan2(y, x)  # rad

    # 5) Sett sammen 2 (opp/ned) for "høyre"
    sol_right_up   = [theta1_right, th2_up,   th3_up]
    sol_right_down = [theta1_right, th2_down, th3_down]

    # 6) "Venstrevendt" => +180° = +pi på BOTH theta1 og theta2
    #   => for opp / ned
    def add_pi(angle):
        return angle + np.pi

    sol_left_up = [
        add_pi(theta1_right),       # base + pi
        add_pi(th2_up),             # skulder + pi
        -th3_up                      # albue 
    ]
    sol_left_down = [
        add_pi(theta1_right),
        add_pi(th2_down),
        -th3_down
    ]

    # Samle alle 4
    solutions_rad = [
        sol_right_up,
        sol_right_down,
        sol_left_up,
        sol_left_down
    ]

    # 7) Konverter til grader & round
    deg_solutions = []
    for sol in solutions_rad:
        sol_deg = [np.rad2deg(a) for a in sol]
        sol_deg_4 = [round(a, 4) for a in sol_deg]
        deg_solutions.append(sol_deg_4)

    return deg_solutions



solutions = find_four_solutions()
print("Her er 4 IK-løsninger (theta1, theta2, theta3) i grader:")
for i, sol in enumerate(solutions, 1):
    # Formater hvert tall til 4 desimaler, som string
    sol_str = [f"{angle:.4f}" for angle in sol]
    print(f"Løsning {i}: {sol_str}")

#3a

import numpy as np

def jacobian(joint_angles, joint_velocities):
    """
    jacobian - calculates the cartesian velocity of the tip (x_dot,y_dot,z_dot)
    for a 3-DOF 'simplified CrustCrawler' manipulator.

    Parameters:
      joint_angles:     np.array([theta1, theta2, theta3])
      joint_velocities: np.array([dtheta1, dtheta2, dtheta3])

    Returns:
      cart_velocities:  np.array([vx, vy, vz])
    
    Also prints Jv (3x3) and Jw (3x3).
    """

    # Ledd-lengder (mm):
    L1 = 100.9
    L2 = 222.1
    L3 = 136.2

    
    joint_angles = np.deg2rad(joint_angles)

    t1, t2, t3 = joint_angles

    # Forkortelser:
    c1, s1 = np.cos(t1), np.sin(t1)
    c2, s2 = np.cos(t2), np.sin(t2)
    c23, s23 = np.cos(t2 + t3), np.sin(t2 + t3)

    r = L2 * c2 + L3 * c23

 # 1) Jv = partiell derivert av (x,y,z)
    Jv = np.array([
        [-s1 * r,           -c1*(L2*s2 + L3*s23),  -L3*c1*s23],
        [ c1 * r,           -s1*(L2*s2 + L3*s23),  -L3*s1*s23],
        [    0,             L2*c2 + L3*c23,        L3*c23     ]
    ])

    # Jw = rotasjonsdel, 3x3
 
    z0 = np.array([0, 0, 1])

    z1 = np.array([s1, -c1, 0])

    z2 = np.array([s1, -c1, 0])

    Jw = np.column_stack([z0, z1, z2])

    # Vis matrisene til skjerm:
    print("Jv (3x3) =")
    print(Jv)
    print("Jw (3x3) =")
    print(Jw)

 
    # 3) multipliser for å få cart. velocity (vx,vy,vz): @ er en matrisemultiplikator for numpy objekter
    cart_velocities = Jv @ joint_velocities

    return cart_velocities

angles = np.array([np.deg2rad(90), np.deg2rad(90+60), np.deg2rad(45)])
dangles = np.array([0.1, 0.05, 0.05])  # rad/s

v_cart = jacobian(angles, dangles)
print("Cartesian velocity (vx, vy, vz):", v_cart)
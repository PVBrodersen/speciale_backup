# Noter for planlægning 

### Møde med Helle Brodersen 19.11.24

Punkter der skal fuldføres

- Kamera kalibrering (2 Arbejdsdage)
- Stereo kamera kalibrering (Kamera extrinsics) (2-3 Arbejdsdage)
  - Hertil skal der testes hvordan en pixel mappes fra ét billede til et andet
- Robot interfacing (1 Arbejdsdag)
  - Opsætning af Rosbag 
- Hardware skal tilsluttes (1 Arbejdsdag)
- Data indsamling (4 Arbejdsdage)
  - Kør i hvert fald 3-4 forskellige steder 
- Data preprocessing (10-12 Arbejdsdage)
  - Mapping af IMU data til de respective billeder 
  - Cropping af billeder så kun IMU indkapsuleret data er i billedet, resten fyldes med NaN 
  - Interpolation fra IMU reading til en Rainbow heat mapping 
- Network design (7 Arbejdsdage)
  - Find optimal struktur 
  - Undersøg mere om status quo 
- Træning (2 Arbejdsdage)
  - Lav træningssæt, validering, og test
  - Dokumentér resultater
- Skriv rapport (10-14 Arbejdsdage i parallelt)
  - Indledning 

#### Nice to have 
  - Implementér netværk på jetson
  - Kør netværket med live billed feed 
  - Lav path planning med dette 
  - Forsøg i markant anerledes miljøer
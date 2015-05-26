# kinectshop

Aktueller Stand unsere Erkennungssoftware. 
Basis ist opencv 2.4.9, da ich einige Probleme mit 2.4.4 hatte. 
Ihr müsst dazu nur die System Umgebungsvariablen an das neue lib Verzeichnis von opencv 2.4.9 anpassen.

Fertig implementiert:

+ Histogrammvergleich
+ SURF Detector

Probleme:
- Histogrammvergleich ist nicht sonderlich robust (Hintergrund stört)

Es Fehlt:
- Erkennung des Rechtecks für einen zuverlässigen Vergleich der Histogramme
  (Implementierung auf Basis der squares.cpp empfohlen)


Benutze Umgebung:
opencv 2.4.9 / x86 / vc12
 / visual studio ultimate 2013


Stand 26.05.2015 Tobias
